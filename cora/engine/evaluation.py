"""Helpers of ``tools/evaluate.py``: which weights, which sets, where the result goes (paper Sec. 4.1, Tables 1 to 4).

Kept in the package so that they are unit-tested; the script only wires them to the command line.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from cora.config import Config, load_config
from cora.engine import checkpoint as ckpt
from cora.engine.evaluator import available_sets

logger = logging.getLogger("cora")


def latest_config_file(run_dir: Path) -> Path:
    """Newest ``config*.yaml`` of a run folder (the configuration of the most recent invocation that changed it)."""
    files = sorted(Path(run_dir).glob("config*.yaml"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no config.yaml in {run_dir}; is it a run folder written by tools/train.py?")
    return files[-1]


def load_run_config(config: Optional[str], run_dir: Optional[str], overrides: Sequence[str]) -> Config:
    """Configuration for an evaluation.

    With ``run_dir`` the run's own saved configuration is used (unless ``config`` is given) and the output
    location is that folder, so the model has exactly the shapes of the checkpoints.
    """
    extra: List[str] = []
    if run_dir:
        folder = Path(run_dir).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"run folder {folder} does not exist")
        config = config or str(latest_config_file(folder))
        extra = [f"paths.output_dir={folder.parent}", f"train.run_name={folder.name}"]
    return load_config(config, extra + list(overrides))


def select_sets(cfg: Config, split: str, named: Optional[Sequence[str]]) -> List[str]:
    """Instruction sets to evaluate.

    Sets named by the user (``--sets``) must exist. Sets that come from ``eval.sets`` are skipped, with a log
    message, when they have no instruction file, as the trainer does.
    """
    folder = cfg.data_dir / "instructions" / split
    if named:
        present, missing = available_sets(cfg, split, named)
        if missing:
            raise FileNotFoundError(f"no {split} instruction files for {missing} under {folder}")
        return list(named)
    present, missing = available_sets(cfg, split, cfg.eval.sets)
    if missing:
        logger.warning("no %s instruction files for %s under %s: skipped (name sets with --sets to require them)",
                       split, missing, folder)
    if not present:
        raise FileNotFoundError(f"none of the evaluation sets {cfg.eval.sets} has {split} instruction files under {folder}")
    return present


def load_weights(model, cfg: Config, checkpoint: Optional[str], stage: Optional[str]) -> Tuple[str, str, Set[str]]:
    """Load the weights to evaluate; returns ``(description, short tag for file names, loaded tensor names)``.

    * ``checkpoint=None, stage=None``: every finished stage of the run, in order (what training ends with).
    * ``stage``: the finished stages up to and including that stage.
    * ``checkpoint``: that file; a ``final.pt`` inside a stage folder is preceded by the finished stages before it.
    """
    names = [s.name for s in cfg.train.stages]
    if checkpoint is None:
        if stage is not None and stage not in names:
            raise ValueError(f"unknown stage {stage!r}; configured stages: {names}")
        loaded, tensors = ckpt.load_finished_stages(model, cfg.run_dir, names, upto=stage)
        if not loaded:
            raise FileNotFoundError(
                f"no finished stage in {cfg.run_dir}; pass --checkpoint FILE, --run-dir of a finished run, or --no-checkpoint"
            )
        if stage is not None and stage not in loaded and not ckpt.is_pruned(cfg.run_dir, stage):
            raise FileNotFoundError(f"stage {stage} is not finished in {cfg.run_dir} (finished: {loaded})")
        return f"{cfg.run_dir} ({'+'.join(loaded)})", loaded[-1], tensors
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint {path} does not exist")
    folder_stage = path.parent.name
    if path.name == ckpt.FINAL and folder_stage in names:
        loaded, tensors = ckpt.load_finished_stages(model, path.parent.parent, names, upto=folder_stage)
        if folder_stage in loaded:
            # Only a stage of the evaluated run gets the plain stage tag: a stage of another run folder must not
            # be filed (and later reported) as this run's own result.
            own = path.parent.parent.resolve() == Path(cfg.run_dir).resolve()
            return str(path), folder_stage if own else f"{path.parent.parent.name}-{folder_stage}", tensors
    state = ckpt.load_checkpoint(path)["model"]
    ckpt.load_state_checked(model, state, str(path))
    return str(path), f"{path.parent.name}-{path.stem}", set(state)


def warn_missing_tensors(model, loaded: Set[str], sets: Sequence[str]) -> List[str]:
    """Log which trainable tensors the loaded weights do not cover; returns the tensor-group names (``anchors`` ...)."""
    missing = [n for n, p in model.named_parameters() if p.requires_grad and n not in loaded]
    groups: Dict[str, int] = {}
    for name in missing:
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + 1
    if groups:
        logger.warning("the loaded weights do not cover %d trainable tensors (%s); they keep their initial values",
                       len(missing), ", ".join(f"{g}: {n}" for g, n in sorted(groups.items())))
        if "anchors" in groups and any(s.startswith("conditional") for s in sets):
            logger.warning("the anchor encoders are untrained but the conditional set is evaluated: "
                           "use weights of a stage that trains them (stage2 or later)")
    return sorted(groups)


def require_query_lengths(dataset, allow_unknown: bool) -> int:
    """Number of ReasonSeg images without a long/short flag; raises unless ``allow_unknown`` (Table 3, ``Test (LQ)``).

    Without the flag the evaluator cannot form the long-query group and the prompt of every query is the long
    one, so a Table 3 run would silently lack half of its numbers.
    """
    unknown = sum(dataset.is_long_query(i) is None for i in range(len(dataset)))
    if unknown and not allow_unknown:
        raise ValueError(
            f"{unknown} of {len(dataset)} ReasonSeg images have no long/short query flag, so the Test (LQ) number "
            f"of Table 3 cannot be computed and every query would get the long prompt. Set paths.reasonseg_manifest "
            f"to a JSON with the flag per image id (or use LISA's original ReasonSeg folder with its .json files); "
            f"--allow-unknown-query-length evaluates anyway (debugging only)."
        )
    return unknown


def table_columns(split: str, results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """ReasonSeg groups under the column names of Table 3: ``Test`` / ``Test (LQ)`` (``Val`` for the val split)."""
    label = split.capitalize()
    columns = {label: results["all"]} if "all" in results else {}
    if "long" in results:
        columns[f"{label} (LQ)"] = results["long"]
    return columns


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", text).strip("-")


def result_filename(name: str, mode: str, tag: str, sets: Optional[Sequence[str]], limit: Optional[int]) -> str:
    """``<name>_<mode>__<weights>__<sets>[__n<limit>].json``: results of other weights, sets or subsets never collide."""
    parts = [f"{name}_{mode}", _slug(tag)]
    if sets:
        parts.append("+".join(sets))
    if limit:
        parts.append(f"n{limit}")
    return "__".join(parts) + ".json"
