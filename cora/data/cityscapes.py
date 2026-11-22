"""Cityscapes adapter (paper Sec. 4.1, supplement Sec. 4).

Image ids are ``"<split>/<city>/<stem>"`` (for example ``"train/aachen/aachen_000000_000019"``). The
image is ``leftImg8bit/<id>_leftImg8bit.png`` and the label ``gtFine/<id>_gtFine_labelIds.png`` whose
labelIds are mapped to the 19 evaluation trainIds (everything else is 255). The UniMatch / SemiVL split
lists name ``..._labelTrainIds.png`` files; only the leading image path of each line is used here.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cora import constants as C
from cora.config import REPO_ROOT
from cora.data.adapters import check_split, check_which, read_id_list

SPLITS_DIR = REPO_ROOT / "assets" / "splits" / "cityscapes"
IMAGE_SUFFIX = "_leftImg8bit.png"
LABEL_SUFFIX = "_gtFine_labelIds.png"
EXPECTED_COUNTS = {  # labeled, unlabeled (train = 2975 images), val = 500 (paper Sec. 4.1 / SemiVL lists)
    "1_30": (100, 2875),
    "1_16": (186, 2789),
    "1_8": (372, 2603),
    "1_4": (744, 2231),
}
EXPECTED_TRAIN, EXPECTED_VAL = 2975, 500
_ID_PATTERN = re.compile(r"^(train|val|test)/([^/\\]+)/([^/\\]+)$")


def labelid_lut() -> np.ndarray:
    """256-entry lookup table labelId -> trainId (255 for the 15 ignored ids and anything unknown)."""
    lut = np.full(256, C.IGNORE_LABEL, dtype=np.uint8)
    for label_id, train_id in C.CITYSCAPES_LABELID_TO_TRAINID.items():
        lut[label_id] = train_id
    return lut


def split_line_to_id(line: str) -> str:
    """``leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png [label path]`` -> image id."""
    image_path = line.split()[0]
    prefix, suffix = "leftImg8bit/", IMAGE_SUFFIX
    if not (image_path.startswith(prefix) and image_path.endswith(suffix)):
        raise ValueError(f"unexpected split-list entry {line!r}: expected leftImg8bit/<split>/<city>/<stem>{suffix}")
    image_id = image_path[len(prefix) : -len(suffix)]
    if not _ID_PATTERN.match(image_id):
        raise ValueError(f"unexpected split-list entry {line!r}: cannot derive an image id")
    return image_id


class CityscapesAdapter:
    """Cityscapes with trainId labels derived from the labelIds files (no labelTrainIds files are needed)."""

    name = "cityscapes"
    class_names = C.CITYSCAPES_CLASSES

    def __init__(self, root: Path, splits_dir: Optional[Path] = None) -> None:
        self.root = Path(root)
        self.splits_dir = Path(splits_dir) if splits_dir is not None else SPLITS_DIR
        self._lut = labelid_lut()
        self._ids: dict = {}

    # ---- ids -----------------------------------------------------------------------------------
    def image_ids(self, split: str) -> list:
        """Sorted ids found on disk under ``leftImg8bit/<split>`` ("test" has no public labels and is rejected)."""
        check_split(split)
        if split not in self._ids:
            folder = self.root / "leftImg8bit" / split
            if not folder.is_dir():
                raise FileNotFoundError(f"Cityscapes folder not found: {folder}")
            files = sorted(folder.glob(f"*/*{IMAGE_SUFFIX}"))
            self._ids[split] = [f"{split}/{f.parent.name}/{f.name[: -len(IMAGE_SUFFIX)]}" for f in files]
        return list(self._ids[split])

    def partition(self, partition: str, which: str) -> list:
        """Ids listed in ``assets/splits/cityscapes/<partition>/<which>.txt`` (labeled | unlabeled)."""
        check_which(which)
        if partition not in C.CITYSCAPES_PARTITIONS:
            raise ValueError(f"unknown partition {partition!r}; expected one of {C.CITYSCAPES_PARTITIONS}")
        return self._read_list(self.splits_dir / partition / f"{which}.txt")

    def val_list(self) -> list:
        """Ids listed in ``val.txt`` (the 500 validation images)."""
        return self._read_list(self.splits_dir / "val.txt")

    @staticmethod
    def _read_list(path: Path) -> list:
        return _cached_list(str(path))[:]

    # ---- files ---------------------------------------------------------------------------------
    @staticmethod
    def _check_id(image_id: str) -> None:
        if not _ID_PATTERN.match(image_id):
            raise ValueError(f"invalid Cityscapes image id {image_id!r}; expected '<split>/<city>/<stem>'")

    def image_path(self, image_id: str) -> Path:
        self._check_id(image_id)
        return self.root / "leftImg8bit" / f"{image_id}{IMAGE_SUFFIX}"

    def label_path(self, image_id: str) -> Path:
        """Path of the labelIds PNG."""
        self._check_id(image_id)
        return self.root / "gtFine" / f"{image_id}{LABEL_SUFFIX}"

    def load_image(self, image_id: str) -> np.ndarray:
        path = self.image_path(image_id)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read Cityscapes image {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_label(self, image_id: str) -> np.ndarray:
        path = self.label_path(image_id)
        label_ids = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if label_ids is None:
            raise FileNotFoundError(f"cannot read Cityscapes label {path}")
        if label_ids.ndim != 2 or label_ids.dtype != np.uint8:
            raise ValueError(f"{path}: expected a single-channel uint8 labelIds image, got {label_ids.dtype} {label_ids.shape}")
        return self._lut[label_ids]

    # ---- verification --------------------------------------------------------------------------
    def verify(self) -> list:
        """Check the dataset against the paper / split lists; returns a list of problem descriptions (empty = ok).

        Checks: 2975 train and 500 val images with labelIds files, ``val.txt`` equal to the validation images,
        the labeled / unlabeled counts of every partition, disjointness and that both lists together are exactly
        the training set, and that every listed image and label exists.
        """
        problems: list = []
        train, val = self.image_ids("train"), self.image_ids("val")
        for name, ids, expected in (("train", train, EXPECTED_TRAIN), ("val", val, EXPECTED_VAL)):
            if len(ids) != expected:
                problems.append(f"{name}: found {len(ids)} images, expected {expected}")
            missing = [i for i in ids if not self.label_path(i).is_file()]
            if missing:
                problems.append(f"{name}: {len(missing)} images without labelIds file (first: {missing[0]})")
        listed_val = self.val_list()
        if set(listed_val) != set(val) or len(listed_val) != len(val):
            problems.append("val.txt does not list exactly the validation images found on disk")
        train_set = set(train)
        for part, (n_lab, n_unlab) in EXPECTED_COUNTS.items():
            labeled, unlabeled = self.partition(part, "labeled"), self.partition(part, "unlabeled")
            if (len(labeled), len(unlabeled)) != (n_lab, n_unlab):
                problems.append(f"{part}: {len(labeled)}/{len(unlabeled)} labeled/unlabeled, expected {n_lab}/{n_unlab}")
            if set(labeled) & set(unlabeled):
                problems.append(f"{part}: labeled and unlabeled lists overlap")
            if len(set(labeled)) != len(labeled) or len(set(unlabeled)) != len(unlabeled):
                problems.append(f"{part}: duplicate ids in a list")
            if set(labeled) | set(unlabeled) != train_set:
                problems.append(f"{part}: labeled + unlabeled are not exactly the training images")
        return problems


@lru_cache(maxsize=64)
def _cached_list(path: str) -> list:
    return [split_line_to_id(line) for line in read_id_list(Path(path))]
