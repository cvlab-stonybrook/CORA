"""Binary masks of the CORA instruction records (see README.txt).

    python masks.py show instructions/val/conditional.jsonl 0
    python masks.py export instructions/val/conditional.jsonl out_dir [--limit N]
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=8)
def load_regions(split: str, image_id: str, objects_root: str = str(ROOT / "objects")):
    """(region map as uint16 array, {region_id: index}, {label_id: [indices]}) of one image."""
    base = Path(objects_root) / split / image_id
    region_map = np.array(Image.open(str(base) + ".png"))
    table = json.loads(Path(str(base) + ".json").read_text())
    by_id = {r["region_id"]: r["index"] for r in table["regions"]}
    by_label = {}
    for r in table["regions"]:
        by_label.setdefault(r["label_id"], []).append(r["index"])
    return region_map, by_id, by_label


def object_mask(ref, split: str, image_id: str, objects_root: str = str(ROOT / "objects")) -> np.ndarray:
    """Boolean (H, W) mask of a target or anchor: its regions, or the whole class when region_ids is empty."""
    region_map, by_id, by_label = load_regions(split, image_id, objects_root)
    if ref["region_ids"]:
        indices = [by_id[r] for r in ref["region_ids"]]
    else:
        indices = by_label.get(ref["label_id"], [])
    return np.isin(region_map, indices)


def record_masks(record, objects_root: str = str(ROOT / "objects")):
    """(target mask, anchor mask or None) of one instruction record."""
    target = object_mask(record["target"], record["split"], record["image_id"], objects_root)
    anchor = None
    if record.get("anchor"):
        anchor = object_mask(record["anchor"], record["split"], record["image_id"], objects_root)
    return target, anchor


def read_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="print one record and the size of its masks")
    show.add_argument("file")
    show.add_argument("line", type=int, help="0-based line of the record")
    export = sub.add_parser("export", help="write the masks of a file as PNG images (0 / 255)")
    export.add_argument("file")
    export.add_argument("out_dir")
    export.add_argument("--limit", type=int, default=None, help="only the first N records")
    args = parser.parse_args()

    if args.command == "show":
        record = next(r for i, r in enumerate(read_records(args.file)) if i == args.line)
        target, anchor = record_masks(record)
        print(json.dumps({k: record[k] for k in ("id", "query", "target", "anchor", "relation")}, indent=2))
        print(f"target mask: {target.shape}, {int(target.sum())} pixels" + (f"; anchor mask: {int(anchor.sum())} pixels" if anchor is not None else ""))
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, record in enumerate(read_records(args.file)):
        if args.limit is not None and i >= args.limit:
            break
        target, anchor = record_masks(record)
        Image.fromarray(target.astype(np.uint8) * 255).save(out / f"{i:06d}_target.png")
        if anchor is not None:
            Image.fromarray(anchor.astype(np.uint8) * 255).save(out / f"{i:06d}_anchor.png")
        count += 1
    print(f"wrote the masks of {count} records to {out}")


if __name__ == "__main__":
    main()
