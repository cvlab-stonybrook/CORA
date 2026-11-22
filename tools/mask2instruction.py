#!/usr/bin/env python
"""Convert plain semantic masks into pseudo reasoning supervision, in one command (paper Sec. 1 contribution,
Sec. 3.4.2 and Fig. 5, supplement Sec. 4).

The paper's "practical guideline for converting plain semantic masks into pseudo reasoning supervision using
off-the-shelf models" is a four-step recipe. This tool runs the steps in order for the dataset selected by
``data.dataset`` (the adapter of that dataset supplies image, semantic mask and class table), calling the
existing tools in-process with identical arguments, so every output lands where the training code expects it:

  1. objects      connected-component regions of every mask         tools/build_objects.py
  2. rule sets    semantic set and templated conditional set        tools/generate_rule_sets.py
  3. query DB     per-class query database D                        tools/generate_rule_sets.py query-db
  4. pseudo-labels (optional) teacher predictions for the          tools/make_pseudo_labels.py
                  unlabeled images; needs ``--teacher``

Step 4 is off by default (the teacher needs a checkpoint). The LLM-written attribute and conditional sets are not
produced here: they come with the provided instruction files. ``--dry-run`` prints the commands of the plan and runs
nothing.

    # Cityscapes, 1/16 partition: objects, semantic + templated sets, shipped query database D
    python tools/mask2instruction.py --partition 1_16 --splits train val
    # additionally the pseudo-labels of the unlabeled images
    python tools/mask2instruction.py --partition 1_16 --teacher unimatch --checkpoint <ckpt> --date 2026-11-19

Another dataset needs an adapter (``cora.data.adapters.DatasetAdapter``) registered in ``get_adapter`` and its class
table in ``cora.constants``; the tool itself is dataset-agnostic.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cora.config import load_config  # noqa: E402

LOG = logging.getLogger("mask2instruction")
TOOLS_DIR = Path(__file__).resolve().parent
STEPS = ("objects", "rule-sets", "query-db", "pseudo")
TEACHERS = ("unimatch", "semivl", "oracle-noise")
_TOOL_MODULES: Dict[str, object] = {}


@dataclass(frozen=True)
class Command:
    """One invocation of an existing tool: ``tool.main(argv)``."""

    step: str
    tool: str
    argv: tuple

    def shell(self) -> str:
        """The command as it would be typed in a shell."""
        return " ".join(["python", f"tools/{self.tool}.py", *map(shlex.quote, self.argv)])


def load_tool(name: str):
    """Import ``tools/<name>.py`` by path (``tools`` is not a package) and cache the module."""
    if name not in _TOOL_MODULES:
        spec = importlib.util.spec_from_file_location(f"cora_tool_{name}", TOOLS_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # worker processes pickle functions of a tool by module name
        spec.loader.exec_module(module)
        _TOOL_MODULES[name] = module
    return _TOOL_MODULES[name]


def _config_args(args: argparse.Namespace) -> List[str]:
    """``--config`` / ``--set`` options shared by every tool."""
    out: List[str] = []
    if args.config:
        out += ["--config", args.config]
    for override in args.overrides:
        out += ["--set", override]
    return out


def _image_selection(args: argparse.Namespace, split: str) -> List[str]:
    """Image selector options common to build_objects / generate_rule_sets."""
    if split == "val":  # --image-list and --partition select training images; val is always every validation image
        return ["--split", "val"]
    if args.image_list:
        return ["--split", "train", "--image-list", str(args.image_list)]
    return ["--partition", args.partition]


def build_plan(args: argparse.Namespace) -> List[Command]:
    """The ordered list of tool invocations for the chosen steps; raises ``SystemExit`` on inconsistent options."""
    steps = set(args.steps)
    if "train" in args.splits and not (args.partition or args.image_list) and steps & {"objects", "rule-sets"}:
        raise SystemExit("the train split needs --partition (such as 1_16) or --image-list")
    if args.teacher and "pseudo" not in steps:
        steps.add("pseudo")
    if "pseudo" in steps:
        if not args.teacher:
            raise SystemExit("the pseudo step needs --teacher " + "|".join(TEACHERS))
        if not args.partition or not args.date:
            raise SystemExit("pseudo-labels need --partition and --date (recorded in the manifest)")
        if args.teacher != "oracle-noise" and not args.checkpoint:
            raise SystemExit(f"--teacher {args.teacher} needs --checkpoint")
    cfg_args = _config_args(args)
    limit = ["--limit", str(args.limit)] if args.limit else []
    plan: List[Command] = []
    if "objects" in steps:
        for split in args.splits:
            plan.append(Command("objects", "build_objects", tuple(
                cfg_args + _image_selection(args, split) + ["--workers", str(args.workers)] + limit)))
    if "rule-sets" in steps:
        for split in args.splits:
            sel = _image_selection(args, split)
            for sub in ("semantic", "templated-conditional"):
                plan.append(Command("rule-sets", "generate_rule_sets", tuple(cfg_args + [sub] + sel + limit)))
    if "query-db" in steps:
        plan.append(Command("query-db", "generate_rule_sets", tuple(
            cfg_args + ["query-db"] + (["--force"] if args.force_query_db else []))))
    if "pseudo" in steps:
        argv = ["--method", args.teacher, "--partition", args.partition, "--date", args.date,
                "--dataset", args.dataset_name] + cfg_args
        if args.checkpoint:
            argv += ["--checkpoint", args.checkpoint]
        if args.gpus:
            argv += ["--gpus", args.gpus]
        elif args.workers > 1:
            argv += ["--workers", str(args.workers)]
        plan.append(Command("pseudo", "make_pseudo_labels", tuple(argv + limit)))
    return plan


def run_command(command: Command, runner: Optional[Callable[[Command], int]] = None) -> int:
    """Run one command in-process; a ``SystemExit`` of the tool is re-raised with the failing step named."""
    LOG.info("[%s] %s", command.step, command.shell())
    if runner is not None:
        return runner(command)
    try:
        return int(load_tool(command.tool).main(list(command.argv)) or 0)
    except SystemExit as exit_:
        if exit_.code in (0, None):
            return 0
        raise SystemExit(f"step '{command.step}' ({command.tool}) stopped: {exit_.code}") from exit_


def run_plan(plan: Sequence[Command], runner: Optional[Callable[[Command], int]] = None) -> None:
    """Run the commands in order and stop at the first failure."""
    for command in plan:
        code = run_command(command, runner)
        if code != 0:
            raise SystemExit(f"step '{command.step}' ({command.tool}) returned exit code {code}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML configuration (default: built-in defaults)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="config override (data.dataset selects the dataset)")
    parser.add_argument("--partition", help="labeled fraction of the training set, such as 1_16")
    parser.add_argument("--image-list", type=Path, help="train images from a text file (one id per line) instead of a partition (val always uses every image)")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train"], help="splits to convert")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=["objects", "rule-sets", "query-db"],
                        help="steps to run, in the recipe's order (default: objects rule-sets query-db)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N images of every selection")
    parser.add_argument("--workers", type=int, default=1, help="processes of build_objects (and CPU shards of the teacher)")
    parser.add_argument("--force-query-db", action="store_true", help="replace an existing query database")
    parser.add_argument("--teacher", choices=TEACHERS, help="pseudo-label method; adds the pseudo step")
    parser.add_argument("--checkpoint", help="teacher checkpoint")
    parser.add_argument("--date", help="date for the pseudo-label manifest, YYYY-MM-DD")
    parser.add_argument("--gpus", help="physical GPU indices for the teacher, e.g. 7")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.workers < 1:
        parser.error("--limit must be >= 0 and --workers >= 1")
    args.dataset_name = load_config(args.config, args.overrides).data.dataset
    return args


def main(argv: Optional[Sequence[str]] = None, runner: Optional[Callable[[Command], int]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plan = build_plan(args)
    if not plan:
        raise SystemExit("nothing to do: the chosen steps produce no commands")
    if args.dry_run:
        for i, command in enumerate(plan, 1):
            print(f"{i}. [{command.step}] {command.shell()}")
        return 0
    run_plan(plan, runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
