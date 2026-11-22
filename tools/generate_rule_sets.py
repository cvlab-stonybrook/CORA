#!/usr/bin/env python
"""Rule-based instruction sets and the query database D (paper Sec. 3.4.2, supplement Sec. 8).

Subcommands (each writes to the paths of ``Config``):

  semantic               class-level semantic set          -> instructions/<split>/semantic.jsonl
  templated-conditional  "Segment the cars on the left of <REF>"-style set
                                                           -> instructions/<split>/conditional_templated.jsonl
  offline                stand-in attribute / conditional sets for pipeline tests (refuses to write next to
                         LLM-written files)                -> instructions/<split>/{attribute,conditional}.jsonl
  query-db               per-class query database D (the shipped assets/query_db/<dataset>.json, validated)
                                                           -> query_db.json
  stats                  per-class count and mean length of stored instruction files (supplement Fig. 2a / 2b)
                                                           -> table on stdout, optional --json / --plot files

The LLM-written attribute and conditional sets are not generated here: they come with the provided instruction files.

Images: ``--image-list FILE`` (one id per line), ``--partition 1_30 [--which labeled|unlabeled]`` (training images
of a partition) or ``--split`` (train = the labeled images of ``data.partition``; val = every validation image).

``--config FILE`` and ``--set KEY=VALUE`` belong before the subcommand:

    python tools/generate_rule_sets.py --set data.partition=1_30 semantic --split train
    python tools/generate_rule_sets.py templated-conditional --split train --anchor-name omit
    python tools/generate_rule_sets.py --set paths.data_dir=/tmp/stand_in offline --split train --sets attribute conditional
    python tools/generate_rule_sets.py query-db
    python tools/generate_rule_sets.py stats --split train --set-name conditional --plot fig2.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cora import constants as C  # noqa: E402
from cora.config import Config, load_config  # noqa: E402
from cora.data.adapters import DatasetAdapter, get_adapter, read_id_list  # noqa: E402
from cora.data.query_db import QueryDB, shipped_path  # noqa: E402
from cora.data.relations import ALL_RELATIONS  # noqa: E402
from cora.data.schema import InstructionRecord, read_jsonl, write_jsonl  # noqa: E402
from cora.instructions.offline import OFFLINE_SETS, OfflineRefusal, build_offline_records, refuse_if_llm_generated  # noqa: E402
from cora.instructions.semantic import build_semantic_records  # noqa: E402
from cora.instructions.stats import instruction_stats, plot_stats, stats_dict, stats_table  # noqa: E402
from cora.instructions.templated_conditional import ANCHOR_NAME_MODES, build_templated_conditional  # noqa: E402

LOG = logging.getLogger("generate_rule_sets")


def select_images(cfg: Config, adapter: DatasetAdapter, args: argparse.Namespace) -> List[str]:
    """Image ids chosen by --image-list, --partition or --split (see the module docstring)."""
    if args.image_list:
        ids = read_id_list(args.image_list)
    elif args.partition:
        if args.split == "val":
            raise SystemExit("--partition selects training images; drop --split val")
        ids = adapter.partition(args.partition, args.which)
    elif args.split == "val":
        ids = adapter.image_ids("val")
    else:
        ids = adapter.partition(cfg.data.partition, "labeled")
    unique = list(dict.fromkeys(ids))  # a repeated id would write every record of that image twice
    if len(unique) < len(ids):
        LOG.warning("%d repeated image id(s) in the selection were dropped", len(ids) - len(unique))
    ids = unique
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        raise SystemExit("no images selected")
    LOG.info("%d images selected (%s split)", len(ids), args.split)
    return ids


def _summary(records: Sequence[InstructionRecord], path: Path) -> dict:
    return {
        "path": str(path),
        "records": len(records),
        "images": len({r.image_id for r in records}),
        "per_class": dict(sorted(Counter(r.target.label_name for r in records).items())),
    }


def cmd_semantic(cfg: Config, args: argparse.Namespace) -> dict:
    adapter = get_adapter(cfg)
    records = list(build_semantic_records(cfg, adapter, args.split, select_images(cfg, adapter, args), min_pixels=args.min_pixels))
    path = cfg.instructions_path(args.split, C.SET_SEMANTIC)
    write_jsonl(path, records)
    return _summary(records, path)


def _relation_thresholds(args: argparse.Namespace) -> dict:
    given = {"near_frac": args.near_frac, "around_frac": args.around_frac, "direction_margin_frac": args.direction_margin_frac,
             "depth_margin_frac": args.depth_margin_frac, "depth_overlap_frac": args.depth_overlap_frac,
             "group_frac": args.group_frac}
    return {k: v for k, v in given.items() if v is not None}


def cmd_templated(cfg: Config, args: argparse.Namespace) -> dict:
    adapter = get_adapter(cfg)
    records = list(build_templated_conditional(
        cfg, adapter, args.split, select_images(cfg, adapter, args), anchor_name=args.anchor_name,
        relations=args.relations, use_anchor_groups=False if args.no_anchor_groups else None, per_image=args.per_image,
        **_relation_thresholds(args),
    ))
    path = cfg.instructions_path(args.split, C.SET_CONDITIONAL, "templated")
    write_jsonl(path, records)
    summary = _summary(records, path)
    summary["per_relation"] = dict(sorted(Counter(r.relation for r in records).items()))
    return summary


def cmd_stats(cfg: Config, args: argparse.Namespace) -> dict:
    """Per-class count and mean length of the stored instruction files of a set (supplement Fig. 2a / 2b)."""
    paths = [args.file] if args.file else cfg.instruction_files(args.split, args.set_name)
    if args.variant:
        paths = [cfg.instructions_path(args.split, args.set_name, args.variant)]
    missing = [p for p in paths if not p.is_file()]
    if missing or not paths:
        raise SystemExit(f"no instruction file to summarise: {missing or 'none found for ' + args.set_name}")
    records = [r for path in paths for r in read_jsonl(path)]
    stats = instruction_stats(records)
    result = {"files": [str(p) for p in paths], **stats_dict(stats)}
    print(stats_table(stats))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.plot:
        result["plot"] = str(plot_stats(stats, args.plot, unit=args.unit))
    return result


def cmd_offline(cfg: Config, args: argparse.Namespace) -> dict:
    for set_name in args.sets:
        refuse_if_llm_generated(cfg, args.split, set_name)
    adapter = get_adapter(cfg)
    records = list(build_offline_records(
        cfg, adapter, args.split, select_images(cfg, adapter, args), args.sets,
        anchor_name=args.anchor_name, **_relation_thresholds(args),
    ))
    result = {}
    for set_name in args.sets:
        subset = [r for r in records if r.set == set_name]
        path = cfg.instructions_path(args.split, set_name)
        write_jsonl(path, subset)
        result[set_name] = _summary(subset, path)
    return result


def cmd_query_db(cfg: Config, args: argparse.Namespace) -> dict:
    path = args.out or cfg.query_db_path
    if path.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"{path} already exists; pass --force to replace it")
    db = QueryDB.load(shipped_path(cfg.data.dataset))
    db.validate()
    if args.dry_run:
        return {"dry_run": True, "path": str(path), "classes": len(db.class_ids())}
    db.save(path)
    return {"path": str(path), "dataset": db.dataset, "queries_per_class": {str(c): len(db.queries(c)) for c in db.class_ids()},
            "meta": db.meta}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--config", default=None, help="YAML configuration (default: built-in defaults)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config override")
    sub = parser.add_subparsers(dest="command", required=True)

    def images(p: argparse.ArgumentParser) -> None:
        p.add_argument("--split", choices=("train", "val"), default="train")
        p.add_argument("--partition", help="a partition such as 1_30 (training images)")
        p.add_argument("--which", choices=("labeled", "unlabeled"), default="labeled", help="partition subset")
        p.add_argument("--image-list", type=Path, help="text file with one image id per line")
        p.add_argument("--limit", type=int, default=0, help="only the first N selected images")

    def geometry(p: argparse.ArgumentParser) -> None:
        p.add_argument("--anchor-name", choices=ANCHOR_NAME_MODES, default="omit",
                       help='"include" writes "above the car <REF>", "omit" writes "above <REF>" (Fig. 2)')
        p.add_argument("--near-frac", type=float, help="relation threshold, fraction of the image diagonal")
        p.add_argument("--around-frac", type=float, help="relation threshold, fraction of the image diagonal")
        p.add_argument("--direction-margin-frac", type=float, help="relation threshold, fraction of the image side")
        p.add_argument("--depth-margin-frac", type=float, help='"in front of" / "behind": bottom-edge difference, fraction of the image height')
        p.add_argument("--depth-overlap-frac", type=float, help='"in front of" / "behind": horizontal overlap, fraction of the narrower box')
        p.add_argument("--group-frac", type=float, help="group anchors: largest gap between the two members, fraction of the image diagonal")

    p = sub.add_parser("semantic", help="class-level semantic set")
    images(p)
    p.add_argument("--min-pixels", type=int, default=1, help="a class needs this many pixels in an image")
    p.set_defaults(func=cmd_semantic)

    p = sub.add_parser("templated-conditional", help='rule-based class-level conditional set ("... on the left of <REF>")')
    images(p)
    geometry(p)
    p.add_argument("--per-image", type=int, help="records per image (default rule_sets.templated_conditional_per_image)")
    p.add_argument("--relations", nargs="+", choices=ALL_RELATIONS,
                   help='relations to use (default: the 2D relations, plus "in front of" / "behind" on Cityscapes)')
    p.add_argument("--no-anchor-groups", action="store_true", help="use single regions as anchors only (default on Cityscapes: pairs too)")
    p.set_defaults(func=cmd_templated)

    p = sub.add_parser("offline", help="stand-in attribute / conditional sets for pipeline tests")
    images(p)
    geometry(p)
    p.add_argument("--sets", nargs="+", choices=OFFLINE_SETS, default=list(OFFLINE_SETS))
    p.set_defaults(func=cmd_offline)

    p = sub.add_parser("query-db", help="per-class query database D (copy and validate assets/query_db/<dataset>.json)")
    p.add_argument("--out", type=Path, help="output file (default Config.query_db_path)")
    p.add_argument("--dry-run", action="store_true", help="report what would happen; write nothing")
    p.add_argument("--force", action="store_true", help="replace an existing database")
    p.set_defaults(func=cmd_query_db)

    p = sub.add_parser("stats", help="per-class count and mean length of a stored instruction set (supplement Fig. 2)")
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--set-name", choices=(C.SET_SEMANTIC, C.SET_ATTRIBUTE, C.SET_CONDITIONAL), default=C.SET_CONDITIONAL,
                   help="instruction set (default conditional, as in Fig. 2); every variant file of the set is read")
    p.add_argument("--variant", help='only this variant, e.g. "templated" (conditional_templated.jsonl)')
    p.add_argument("--file", type=Path, help="read this JSONL file instead")
    p.add_argument("--json", type=Path, help="also write the statistics as JSON")
    p.add_argument("--plot", type=Path, help="also draw Fig. 2a / 2b into this PNG")
    p.add_argument("--unit", choices=("words", "chars"), default="words", help="length unit of the plot")
    p.set_defaults(func=cmd_stats)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.overrides)
    try:
        result = args.func(cfg, args)
    except OfflineRefusal as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
