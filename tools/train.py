#!/usr/bin/env python
"""Train CORA: the three stages of paper Sec. 3.3 in one run (finished stages are skipped on re-runs).

One GPU (the paper's batch size 8 = 4 accumulation steps of 2 images):
    python tools/train.py --config configs/cityscapes/1_16.yaml train.grad_accum_steps=4
Four GPUs (the paper's batch size 8 = 2 images per device):
    python -m torch.distributed.run --nproc_per_node=4 tools/train.py --config configs/cityscapes/1_16.yaml
Options can be overridden after the config (before or after the other options):
    train.run_name=my_run train.iters_per_stage=100

Before the model is built the run is checked: stage order, instruction files, pseudo-labels, query database, and
that a finished or interrupted stage of the run folder was produced with the same settings. ``--check`` does
only that and exits. A run stopped by Ctrl-C or SIGTERM saves its state and resumes from it when started again.
Runs that differ only in their later stages can share the first ones with ``--init-stages-from <run folder>``.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cora.config import load_config  # noqa: E402
from cora.engine.preflight import preflight  # noqa: E402
from cora.engine.trainer import Trainer, TrainingInterrupted  # noqa: E402
from cora.model.build import build_tokenizer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML configuration file (defaults are used when omitted)")
    parser.add_argument("--stages", help="comma-separated stage names to run (default: all); earlier stages must be finished")
    parser.add_argument("--check", action="store_true", help="check stages, data and settings without building the model, then exit")
    parser.add_argument("--allow-config-change", action="store_true",
                        help="continue a run folder although the settings differ from those of its checkpoints")
    parser.add_argument("--init-stages-from", metavar="RUN_DIR",
                        help="link in the finished stages of another run folder that were trained with identical settings")
    parser.add_argument("--prune-checkpoints", action="store_true",
                        help="after the last stage, delete the final.pt of earlier stages that it supersedes (saves about 1 GB each)")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. train.lr=1e-5")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_intermixed_args(argv)
    cfg = load_config(args.config, args.overrides)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    init_from = Path(args.init_stages_from) if args.init_stages_from else None
    log = logging.getLogger("cora")
    if args.check:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _, spec = build_tokenizer(cfg)
        # The process count is part of a stage's settings (effective batch size): under torchrun it is WORLD_SIZE.
        plan = preflight(cfg, spec, stages, world_size=int(os.environ.get("WORLD_SIZE", "1")),
                         allow_config_change=args.allow_config_change, init_stages_from=init_from, log=log)
        print(f"check passed: finished {plan.finished}, to train {plan.to_run}, validation sets {plan.eval_sets}")
        return
    try:
        Trainer(cfg, stages, allow_config_change=args.allow_config_change, init_stages_from=init_from,
                prune_checkpoints=args.prune_checkpoints).fit()
    except TrainingInterrupted as exc:
        log.error(str(exc))
        sys.exit(exc.exit_code)
    except Exception:
        log.exception("training failed")  # also lands in <run_dir>/train.log
        raise


if __name__ == "__main__":
    main()
