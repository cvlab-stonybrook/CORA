#!/usr/bin/env python
"""Check the Cityscapes folder and print per-class pixel and region statistics of a partition.

Verifies the layout and counts the paper relies on (2975 train / 500 val images with labelIds files, the
UniMatch / SemiVL partition lists 100/2875, 186/2789, 372/2603, 744/2231 labeled/unlabeled, val.txt = 500)
and then reports, for the images of one partition, how many pixels, images and connected-component regions
(supplement Sec. 4) every one of the 19 classes has. Nothing is written to the dataset folder.

    python tools/prepare_cityscapes.py --partition 1_30 --which labeled --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cora import constants as C  # noqa: E402
from cora.config import Config  # noqa: E402
from cora.data.cityscapes import CityscapesAdapter  # noqa: E402
from cora.data.objects import count_regions, extract_objects  # noqa: E402

LOG = logging.getLogger("prepare_cityscapes")
_STATE: dict = {}


def _init_worker(root: str, connectivity: int, min_area: int) -> None:
    _STATE.update(adapter=CityscapesAdapter(Path(root)), connectivity=connectivity, min_area=min_area)


def image_statistics(image_id: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """(pixels per class, regions per class, total pixels) of one image."""
    adapter: CityscapesAdapter = _STATE["adapter"]
    label = adapter.load_label(image_id)
    pixels = np.bincount(label.ravel(), minlength=256)[: len(C.CITYSCAPES_CLASSES)]
    objects = extract_objects(image_id, label, adapter.class_names,
                              connectivity=_STATE["connectivity"], min_area=_STATE["min_area"])
    regions = count_regions(objects)
    return pixels, np.array([regions.get(c, 0) for c in range(len(C.CITYSCAPES_CLASSES))]), int(label.size)


def format_table(pixels: np.ndarray, regions: np.ndarray, images_with: np.ndarray, n_images: int, ignored: int) -> str:
    total = int(pixels.sum()) + ignored
    lines = [f"{'class':<14}{'id':>3}{'pixels':>16}{'pixel %':>9}{'images':>8}{'regions':>9}{'regions/img':>12}"]
    for c, name in enumerate(C.CITYSCAPES_CLASSES):
        per_image = regions[c] / images_with[c] if images_with[c] else 0.0
        lines.append(f"{name:<14}{c:>3}{int(pixels[c]):>16}{100 * pixels[c] / total:>9.2f}{int(images_with[c]):>8}"
                     f"{int(regions[c]):>9}{per_image:>12.2f}")
    lines.append(f"{'ignored (255)':<14}{'':>3}{ignored:>16}{100 * ignored / total:>9.2f}")
    lines.append(f"{n_images} images, {int(regions.sum())} regions")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(cfg.paths.cityscapes_root), help="Cityscapes root")
    parser.add_argument("--partition", default="1_30", choices=C.CITYSCAPES_PARTITIONS)
    parser.add_argument("--which", choices=("labeled", "unlabeled", "both"), default="labeled")
    parser.add_argument("--limit", type=int, default=0, help="only the first N images")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=cfg.data.connectivity)
    parser.add_argument("--min-area", type=int, default=cfg.data.min_region_area)
    parser.add_argument("--json-out", type=Path, help="write the statistics as JSON (must be outside the dataset folder)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.root.resolve()
    if args.json_out and (args.json_out.resolve() == root or root in args.json_out.resolve().parents):
        raise SystemExit("--json-out must not be inside the Cityscapes folder")
    adapter = CityscapesAdapter(root)
    problems = adapter.verify()
    for problem in problems:
        LOG.error(problem)
    if problems:
        return 1
    LOG.info("layout ok: %d train / %d val images; partitions %s match the paper lists", len(adapter.image_ids("train")),
             len(adapter.image_ids("val")), ", ".join(C.CITYSCAPES_PARTITIONS))

    parts = ["labeled", "unlabeled"] if args.which == "both" else [args.which]
    ids: List[str] = [i for which in parts for i in adapter.partition(args.partition, which)]
    if args.limit:
        ids = ids[: args.limit]
    initargs = (str(root), args.connectivity, args.min_area)
    if args.workers <= 1:
        _init_worker(*initargs)
        results = [image_statistics(i) for i in ids]
    else:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=initargs) as pool:
            results = list(pool.map(image_statistics, ids, chunksize=8))
    pixels = np.sum([r[0] for r in results], axis=0)
    regions = np.sum([r[1] for r in results], axis=0)
    images_with = np.sum([r[1] > 0 for r in results], axis=0)
    ignored = sum(r[2] for r in results) - int(pixels.sum())
    print(f"partition {args.partition} ({args.which}), connectivity {args.connectivity}, min area {args.min_area}")
    print(format_table(pixels, regions, images_with, len(ids), ignored))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "partition": args.partition, "which": args.which, "images": len(ids), "ignored_pixels": ignored,
            "classes": {n: {"pixels": int(pixels[c]), "images": int(images_with[c]), "regions": int(regions[c])}
                        for c, n in enumerate(C.CITYSCAPES_CLASSES)}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
