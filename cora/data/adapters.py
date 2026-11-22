"""Dataset adapter: one uniform view of Cityscapes (paper Sec. 4.1).

An adapter exposes image ids, the labeled / unlabeled partitions of the semi-supervised protocol
(Sec. 4.2), RGB images and semantic label maps. Label maps are ``uint8`` class ids with
:data:`cora.constants.IGNORE_LABEL` (255) for void / background pixels. Everything downstream (objects,
instruction generation, datasets) talks to this protocol only.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from cora import constants as C
from cora.config import Config

SPLITS = ("train", "val")
WHICH = ("labeled", "unlabeled")


@runtime_checkable
class DatasetAdapter(Protocol):
    """Interface implemented by :class:`CityscapesAdapter`."""

    name: str
    class_names: tuple
    root: Path

    def image_ids(self, split: str) -> list:
        """All image ids of ``split`` ("train" | "val"), sorted."""

    def partition(self, partition: str, which: str) -> list:
        """Ids of the ``labeled`` or ``unlabeled`` training images of a partition such as "1_30"."""

    def image_path(self, image_id: str) -> Path:
        """Path of the RGB image file."""

    def load_image(self, image_id: str) -> np.ndarray:
        """(H, W, 3) uint8 RGB image."""

    def load_label(self, image_id: str) -> np.ndarray:
        """(H, W) uint8 class ids; 255 = void / background."""


def check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


def check_which(which: str) -> None:
    if which not in WHICH:
        raise ValueError(f"unknown partition subset {which!r}; expected one of {WHICH}")


def read_id_list(path: Path) -> list:
    """Read a text file with one id per line (blank lines ignored; a final newline is optional)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"list file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temporary file in the same folder and an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def get_adapter(cfg: Config) -> DatasetAdapter:
    """Adapter of ``cfg.data.dataset`` with the dataset root taken from ``cfg.paths``."""
    if cfg.data.dataset == "cityscapes":
        from cora.data.cityscapes import CityscapesAdapter

        return CityscapesAdapter(cfg.resolve(cfg.paths.cityscapes_root))
    raise ValueError(f"unknown dataset {cfg.data.dataset!r}; expected one of {C.DATASETS}")
