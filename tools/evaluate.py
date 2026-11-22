#!/usr/bin/env python
"""Evaluate a trained CORA model with cIoU (paper Sec. 4.1) and gIoU.

Validation instruction sets of the configured dataset (Tables 1, 2, 4):
    python tools/evaluate.py --run-dir outputs/my_run
ReasonSeg test set, overall and long-query subset (Table 3; needs paths.reasonseg_manifest or LISA's folder):
    python tools/evaluate.py --run-dir outputs/my_run --reasonseg test
The model before any CORA training (a baseline row):
    python tools/evaluate.py --config configs/cityscapes/1_16.yaml --no-checkpoint

``--run-dir`` uses the configuration the run was trained with. By default the weights are those of all finished
stages of the run; ``--stage`` stops at an earlier stage and ``--checkpoint`` names a file. The result goes to
``<run_dir>/eval/<dataset>_<split>_<mode>__<weights>__<sets>[__n<limit>].json`` with the sets, the sample count,
the weights and the configuration, so results of different weights, sets or subsets never overwrite each other.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from cora.config import to_dict  # noqa: E402
from cora.data.collate import make_collate  # noqa: E402
from cora.data.datasets import EvalDataset  # noqa: E402
from cora.engine import dist as D  # noqa: E402
from cora.engine import evaluation as EV  # noqa: E402
from cora.engine.evaluator import evaluate, format_results, shard_indices  # noqa: E402
from cora.model.build import build_model, build_tokenizer  # noqa: E402


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML configuration (default with --run-dir: the run's saved configuration)")
    parser.add_argument("--run-dir", help="run folder written by tools/train.py: its configuration and checkpoints are used")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", help="final.pt or last.pt to evaluate (a stage's final.pt is preceded by the earlier stages)")
    weights.add_argument("--stage", help="evaluate the weights after this stage (default: after the last finished stage)")
    weights.add_argument("--no-checkpoint", action="store_true", help="evaluate the model before any CORA training")
    parser.add_argument("--sets", nargs="*", help="instruction sets to evaluate; each must exist (default: eval.sets of the config, missing ones are skipped)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--reasonseg", choices=["test", "val"], help="evaluate on ReasonSeg instead of the dataset's own sets")
    parser.add_argument("--allow-unknown-query-length", action="store_true",
                        help="ReasonSeg: evaluate although the long/short flag is unknown (no Test (LQ) number; debugging only)")
    parser.add_argument("--mode", choices=["teacher_forced", "generate"])
    parser.add_argument("--max-samples", type=positive_int, help="evaluate an evenly spread subset of this size (default: eval.max_samples)")
    parser.add_argument("--output", help="JSON file for the results (default: see above)")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. eval.batch_size=2")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_intermixed_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = EV.load_run_config(args.config, args.run_dir, args.overrides)
    distributed, rank, world_size, device = D.init_distributed()

    # Everything that can be checked without the model is checked first.
    tokenizer, spec = build_tokenizer(cfg)
    sets = None
    if args.reasonseg:
        from cora.data.reasonseg import ReasonSegDataset

        dataset = ReasonSegDataset(cfg, spec, split=args.reasonseg)
        EV.require_query_lengths(dataset, args.allow_unknown_query_length)
        name, split = f"reasonseg_{args.reasonseg}", args.reasonseg
    else:
        sets = EV.select_sets(cfg, args.split, args.sets)
        dataset = EvalDataset(cfg, spec, sets, split=args.split)
        name, split = f"{cfg.data.dataset}_{args.split}", args.split
    limit = args.max_samples if args.max_samples is not None else cfg.eval.max_samples
    mode = args.mode or cfg.eval.mode

    model = build_model(cfg, spec).to(device)
    if args.no_checkpoint:
        source, tag, loaded = "none", "base", set()
    else:
        source, tag, loaded = EV.load_weights(model, cfg, args.checkpoint, args.stage)
        EV.warn_missing_tensors(model, loaded, sets or [])
    model.eval()

    indices = shard_indices(len(dataset), rank, world_size, limit)
    n_evaluated = len(shard_indices(len(dataset), 0, 1, limit))
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.eval.batch_size, shuffle=False,
                        num_workers=min(4, cfg.data.num_workers),
                        collate_fn=make_collate(tokenizer, spec, cfg, generation=mode == "generate"))
    results = evaluate(model, loader, device, mode, cfg.eval.max_new_tokens)

    if rank == 0:
        print(f"weights: {source}\nevaluation: {name} ({mode}), {n_evaluated} of {len(dataset)} samples")
        print(format_results(results))
        report = {
            "weights": source, "weights_tag": tag, "dataset": name, "mode": mode, "sets": sets, "max_samples": limit,
            "samples_evaluated": n_evaluated, "samples_total": len(dataset), "results": results,
            "config": {k: v for k, v in to_dict(cfg).items() if k != "rule_sets"},
        }
        if args.reasonseg:
            report["table_columns"] = EV.table_columns(split, results)
            print("Table 3 columns: " + ", ".join(f"{k} = {v['ciou']:.2f}" for k, v in report["table_columns"].items()))
        out = Path(args.output) if args.output else cfg.run_dir / "eval" / EV.result_filename(name, mode, tag, sets, limit)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"saved to {out}")
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
