#!/usr/bin/env python
"""Build and store the connected-component objects of a split or a partition (supplement Sec. 4).

Every image of the selection gets ``<data_dir>/<dataset>/objects/<split>/<image_id>.{png,json}``
(see ``cora.data.objects``). Existing files are reused unless ``--force`` is given. The tool reports how
many regions each class has.

    python tools/build_objects.py --partition 1_30 --which labeled --workers 8
    python tools/build_objects.py --split val --workers 8 --set data.dataset=cityscapes
    python tools/build_objects.py --split train --image-list ids.txt --set paths.data_dir=/tmp/cora_data
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cora.config import Config, load_config  # noqa: E402
from cora.data.adapters import DatasetAdapter, get_adapter, read_id_list  # noqa: E402
from cora.data.objects import count_regions, ensure_objects, extract_objects, save_objects  # noqa: E402

LOG = logging.getLogger("build_objects")
_STATE: dict = {}


def _init_worker(cfg: Config, force: bool) -> None:
    _STATE["cfg"], _STATE["adapter"], _STATE["force"] = cfg, get_adapter(cfg), force


def build_one(item: Tuple[str, str]) -> Dict[int, int]:
    """Create (or load) the objects of one (split, image id); returns the region count per class id."""
    split, image_id = item
    cfg, adapter = _STATE["cfg"], _STATE["adapter"]
    if _STATE["force"]:
        objects = extract_objects(image_id, adapter.load_label(image_id), adapter.class_names,
                                  connectivity=cfg.data.connectivity, min_area=cfg.data.min_region_area)
        save_objects(objects, cfg.objects_dir(split))
    else:
        objects = ensure_objects(cfg, adapter, split, image_id)
    return count_regions(objects)


def select_items(adapter: DatasetAdapter, args: argparse.Namespace) -> List[Tuple[str, str]]:
    """(split, image id) pairs chosen by --image-list, --partition or --split."""
    if args.partition and args.split not in (None, "train"):
        raise SystemExit(f"--partition selects training images; it cannot be combined with --split {args.split}")
    if args.partition and args.image_list:
        raise SystemExit("--partition and --image-list select images in different ways; give only one")
    if args.image_list:
        ids, split = read_id_list(args.image_list), args.split or "train"
    elif args.partition:
        split = "train"
        parts = ["labeled", "unlabeled"] if args.which == "both" else [args.which]
        ids = [i for which in parts for i in adapter.partition(args.partition, which)]
    elif args.split:
        split, ids = args.split, adapter.image_ids(args.split)
    else:
        raise SystemExit("choose --split, --partition or --image-list")
    if args.limit:
        ids = ids[: args.limit]
    return [(split, i) for i in ids]


def summarize(adapter: DatasetAdapter, per_image: Sequence[Dict[int, int]]) -> dict:
    """Totals per class: regions, images containing the class, mean regions per image containing it."""
    rows = {}
    for class_id, name in enumerate(adapter.class_names):
        counts = [c[class_id] for c in per_image if c.get(class_id)]
        rows[name] = {"class_id": class_id, "regions": sum(counts), "images": len(counts),
                      "regions_per_image": (sum(counts) / len(counts)) if counts else 0.0}
    return {"images": len(per_image), "regions": sum(r["regions"] for r in rows.values()), "classes": rows}


def format_report(summary: dict) -> str:
    lines = [f"{'class':<16}{'id':>4}{'regions':>10}{'images':>8}{'regions/img':>13}"]
    for name, row in summary["classes"].items():
        lines.append(f"{name:<16}{row['class_id']:>4}{row['regions']:>10}{row['images']:>8}{row['regions_per_image']:>13.2f}")
    lines.append(f"{'total':<16}{'':>4}{summary['regions']:>10}{summary['images']:>8}")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML configuration (default: built-in defaults)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config override")
    parser.add_argument("--split", choices=("train", "val"), help="all images of a split")
    parser.add_argument("--partition", help="a partition such as 1_30 (training images)")
    parser.add_argument("--which", choices=("labeled", "unlabeled", "both"), default="labeled", help="partition subset")
    parser.add_argument("--image-list", type=Path, help="text file with one image id per line (split from --split, default train)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N selected images")
    parser.add_argument("--workers", type=int, default=1, help="processes")
    parser.add_argument("--force", action="store_true", help="re-extract and overwrite existing objects")
    parser.add_argument("--report", type=Path, help="write the per-class counts as JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.overrides)
    adapter = get_adapter(cfg)
    items = select_items(adapter, args)
    if not items:
        raise SystemExit("no images selected")
    LOG.info("%d images -> %s", len(items), cfg.data_dir / "objects")
    if args.workers <= 1:
        _init_worker(cfg, args.force)
        results = [build_one(item) for item in items]
    else:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(cfg, args.force)) as pool:
            results = list(pool.map(build_one, items, chunksize=8))
    summary = summarize(adapter, results)
    print(format_report(summary))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
