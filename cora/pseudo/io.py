"""On-disk interchange format of pseudo-labels (paper Sec. 4.1, Fig. 3 "pretrained semi-supervised segmentation").

The teachers (UniMatch, SemiVL) run offline; CORA reads what they wrote. For every image id ``<id>``
(which may contain ``/``, e.g. ``train/aachen/aachen_000000_000019``) the folder holds

* ``<id>_label.png``  single-channel uint8 class ids (255 = void),
* ``<id>_conf.png``   single-channel uint8 confidence, ``round(confidence * 255)``,

plus one ``manifest.json`` per folder. A float confidence map would need about 12 GB per Cityscapes
partition, so it is stored with 8 bits. A 16-bit map was measured on UniMatch 1/16 predictions: its PNG is
about 6.5 times larger (about 1.0 MB against 0.16 MB per image, 3 GB against 0.5 GB per partition), which the
shared disk cannot hold for 2 teachers x 4 partitions. The price of 8 bits is a resolution of 1/255 in the
threshold of the Sec. 3.2.1 / supplement Sec. 2 confidence weight: the dequantised value differs from the original
by at most 0.5 / 255, so a pixel with ``c`` within 0.5 / 255 of the threshold ``t`` can fall on the wrong side.
The default ``t = 0.9`` is exact (0.9 * 255 = 229.5 lies on a rounding boundary: 0 differing pixels in 16.8 M
measured). For any other ``t`` the effective cut-off is :func:`effective_threshold` (for ``t = 0.95``:
0.951, which dropped 560 to 800 of 2.1 M pixels per image, 0.03 %).

Files are written atomically (temporary file + ``os.replace``); the confidence file is replaced before the
label file, so a label file that exists always comes with a complete confidence file.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple

import cv2
import numpy as np

LABEL_SUFFIX = "_label.png"
CONF_SUFFIX = "_conf.png"
MANIFEST_NAME = "manifest.json"


def _check_image_id(image_id: str) -> PurePosixPath:
    """Reject ids that would escape the pseudo-label folder or that share files with another id.

    An id must be in normal form (no empty or ``.`` parts, no trailing ``/``), so that two different ids never
    map to the same files (``a//b``, ``a/./b`` and ``a/b`` would).
    """
    if not isinstance(image_id, str) or not image_id:
        raise ValueError(f"image id must be a non-empty string, got {image_id!r}")
    posix = PurePosixPath(image_id)
    if posix.is_absolute() or ".." in posix.parts or "\\" in image_id:
        raise ValueError(f"invalid image id {image_id!r}: must be a relative '/'-separated path without '..'")
    if posix.as_posix() != image_id or not posix.name:
        raise ValueError(f"invalid image id {image_id!r}: must be a normalised '/'-separated path with a file name "
                         "(no '.', empty parts or trailing '/')")
    return posix


def prediction_paths(pseudo_dir: Path, image_id: str) -> Tuple[Path, Path]:
    """(label path, confidence path) of an image id."""
    posix = _check_image_id(image_id)
    base = Path(pseudo_dir) / posix.parent
    return base / f"{posix.name}{LABEL_SUFFIX}", base / f"{posix.name}{CONF_SUFFIX}"


def quantize_confidence(confidence: np.ndarray) -> np.ndarray:
    """Float confidence in [0, 1] -> uint8 ``round(confidence * 255)``."""
    conf = np.asarray(confidence)
    if not np.issubdtype(conf.dtype, np.floating):
        raise TypeError(f"confidence must be a float array, got {conf.dtype}")
    if conf.size and (not np.isfinite(conf).all() or conf.min() < 0.0 or conf.max() > 1.0):
        raise ValueError("confidence values must lie in [0, 1]")
    return np.rint(conf.astype(np.float32) * 255.0).astype(np.uint8)


def effective_threshold(threshold: float) -> float:
    """Confidence ``c`` from which a pixel passes ``dequantised(c) >= threshold`` in the 8-bit format.

    A pixel passes when its stored value ``q = round(255 c)`` satisfies ``q >= ceil(255 t)``, that is when
    ``c >= (ceil(255 t) - 0.5) / 255``. The result differs from ``threshold`` by less than 0.5 / 255 and equals
    it when ``255 * threshold + 0.5`` is an integer (the default 0.9).
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    return (math.ceil(round(threshold * 255.0, 9)) - 0.5) / 255.0


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp_name, 0o644)  # mkstemp creates 0600; other processes read the folder
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _encode_png(array: np.ndarray, what: str) -> bytes:
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError(f"PNG encoding of the {what} failed")
    return buffer.tobytes()


def write_prediction(pseudo_dir: Path, image_id: str, label: np.ndarray, confidence: Optional[np.ndarray]) -> None:
    """Write the label map and (optionally) the confidence map of one image.

    ``label`` is (H, W) uint8; ``confidence`` is a float (H, W) array in [0, 1] or None. Without a
    confidence map a stale confidence file of an earlier run is removed, so that the folder never holds a
    label map with the confidence of another prediction.
    """
    label = np.asarray(label)
    if label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError(f"label must be a 2-D uint8 array, got {label.dtype} with shape {label.shape}")
    label_path, conf_path = prediction_paths(pseudo_dir, image_id)
    if confidence is None:
        conf_path.unlink(missing_ok=True)
    else:
        confidence = np.asarray(confidence)
        if confidence.shape != label.shape:
            raise ValueError(f"confidence shape {confidence.shape} differs from label shape {label.shape}")
        _atomic_write_bytes(conf_path, _encode_png(quantize_confidence(confidence), "confidence map"))
    _atomic_write_bytes(label_path, _encode_png(label, "label map"))


def _read_gray(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"pseudo-label file not found: {path}")
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None or array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(f"{path}: expected a single-channel 8-bit PNG")
    return array


def read_prediction(pseudo_dir: Path, image_id: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read (label uint8 (H, W), confidence float32 in [0, 1] or None when no confidence file exists)."""
    label_path, conf_path = prediction_paths(pseudo_dir, image_id)
    label = _read_gray(label_path)
    if not conf_path.is_file():
        return label, None
    conf = _read_gray(conf_path)
    if conf.shape != label.shape:
        raise ValueError(f"{conf_path}: shape {conf.shape} differs from the label map {label.shape}")
    return label, conf.astype(np.float32) / 255.0


def has_prediction(pseudo_dir: Path, image_id: str) -> bool:
    """True when the label file of the image exists (the label file is written last, so it is complete)."""
    return prediction_paths(pseudo_dir, image_id)[0].is_file()


def write_manifest(pseudo_dir: Path, info: dict) -> None:
    """Write ``manifest.json`` (method, partition, checkpoint hash, window settings, date, counts, ...)."""
    text = json.dumps(info, indent=2, sort_keys=True, default=_json_default) + "\n"
    _atomic_write_bytes(Path(pseudo_dir) / MANIFEST_NAME, text.encode("utf-8"))


def read_manifest(pseudo_dir: Path) -> dict:
    path = Path(pseudo_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no manifest in {pseudo_dir}; was the pseudo-label folder written by tools/make_pseudo_labels.py?")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")
