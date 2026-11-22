#!/usr/bin/env python
"""Unpack a ReasonSeg split from its parquet copy into single files for training.

    python tools/prepare_reasonseg.py --split train

Reads ``<paths.reasonseg_root>/reasonseg_<split>/data/*.parquet`` (or LISA's folder layout) through
``cora.data.reasonseg.ReasonSegDataset`` and writes ``<paths.data_dir>/reasonseg/<split>/``:

* ``<image id>.jpg``   the image, in the orientation its mask is drawn in
* ``<image id>.png``   the mask: 0 background, 1 target, 255 ignore
* ``index.json``       ``[{"image_id", "text", "is_sentence"}, ...]`` (``is_sentence`` is LISA's long-query flag,
                       taken from the JSON files or from ``paths.reasonseg_manifest``; null when unknown)

Training draws from this folder when ``data.reasonseg_train_share`` is above zero. The parquet copy holds one
query per image (LISA's JSON files hold several phrasings for some images). The folder is written under a
temporary name and renamed, so an interrupted run leaves nothing half done.

Images whose stored mask does not have the shape of the image in its displayed orientation are left out and
listed: in the train parquet copy 7 of the 239 images carry a rotating EXIF tag while their masks were drawn on an
unrotated canvas, so those masks do not lie on their objects in either orientation.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cora.config import load_config  # noqa: E402
from cora.data.reasonseg import ReasonSegDataset  # noqa: E402
from cora.model.build import build_tokenizer  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML configuration (default: built-in defaults)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config override")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--force", action="store_true", help="replace an existing folder")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    out = cfg.reasonseg_train_dir.parent / args.split
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to replace it")
    _, spec = build_tokenizer(cfg)
    dataset = ReasonSegDataset(cfg, spec, split=args.split)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    index, skipped = [], []
    for i, image_id in enumerate(dataset.image_ids):
        try:
            image, mask, text, is_long = dataset._load(i)
        except ValueError as err:
            if "differs from image shape" not in str(err):
                raise
            skipped.append(image_id)
            continue
        if not set(np.unique(mask).tolist()) <= {0, 1, 255}:
            raise ValueError(f"{image_id}: unexpected mask values {np.unique(mask).tolist()}")
        if not cv2.imwrite(str(tmp / f"{image_id}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"cannot write {tmp / f'{image_id}.jpg'}")
        if not cv2.imwrite(str(tmp / f"{image_id}.png"), mask):
            raise OSError(f"cannot write {tmp / f'{image_id}.png'}")
        index.append({"image_id": image_id, "text": text, "is_sentence": is_long})
    if not index:
        raise SystemExit("no usable image")
    (tmp / "index.json").write_text(json.dumps(index, indent=1))
    (tmp / "skipped.json").write_text(json.dumps(skipped, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    known = [e["is_sentence"] for e in index if e["is_sentence"] is not None]
    print(json.dumps({"folder": str(out), "images": len(index), "long_queries": sum(known),
                      "short_queries": len(known) - sum(known), "flag_unknown": len(index) - len(known),
                      "skipped_mask_in_another_orientation": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
