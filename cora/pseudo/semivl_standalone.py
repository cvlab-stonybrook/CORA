#!/usr/bin/env python
"""SemiVL teacher as a standalone script for SemiVL's own legacy environment (paper Sec. 4.1).

SemiVL (google-research/semivl) needs python 3.8, torch 1.10.1+cu113, mmcv-full 1.4.4, mmsegmentation 0.24.0,
timm 0.6.5 and the OpenAI clip package; none of this can live in the CORA environment. This script therefore
imports nothing from ``cora``: it builds the model from the released ``config.yaml``, loads ``best.pth``,
runs the sliding window of the Cityscapes protocol (crop 801, stride 534, unpadded edge windows, window-count
normalised softmax; identical to ``cora/pseudo/sliding_window.py``) and writes the interchange format of
``cora/pseudo/io.py`` (``<id>_label.png`` uint8 class ids, ``<id>_conf.png`` uint8 ``round(confidence * 255)``)
with a small writer of its own. ``tools/make_pseudo_labels.py --method semivl`` starts it, one process per GPU.

Run it with SemiVL's python, from anywhere (it changes into the SemiVL checkout, whose config paths are
relative):

    $CORA_ROOT/envs/semivl/bin/python semivl_standalone.py --semivl-dir $CORA_ROOT/third_party/semivl \\
        --checkpoint $CORA_ROOT/pretrained/teachers/semivl/semivl_cs_1_16/best.pth \\
        --cityscapes-root <cityscapes> --id-file ids.txt --out-dir <folder> --result-json result.json [--evaluate]

``ids.txt`` lists image ids ``<split>/<city>/<stem>`` (image ``leftImg8bit/<id>_leftImg8bit.png``).
The checkpoint is a third-party pickle: it is read with ``weights_only=True`` when torch supports it and with a
restricted unpickler (allow-list: ``OrderedDict``, tensor rebuild function, storage classes) on torch 1.10,
which has no ``weights_only``.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import os
import pickle
import sys
import tempfile
import time
import types
from pathlib import Path, PurePosixPath

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CROP = 801
DEFAULT_STRIDE = 534  # int(801 * 2 / 3)
IGNORE = 255
NUM_CLASSES = 19
LABEL_SUFFIX = "_label.png"
CONF_SUFFIX = "_conf.png"

# Cityscapes labelId -> trainId (official cityscapesScripts); every other id is 255.
_LABELID_TO_TRAINID = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                       26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


# ----------------------------------------------------------------------------------------------
# Safe checkpoint and config loading
# ----------------------------------------------------------------------------------------------
_ALLOWED_GLOBALS = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2")}
_STORAGE_NAMES = {"FloatStorage", "DoubleStorage", "HalfStorage", "BFloat16Storage", "LongStorage", "IntStorage",
                  "ShortStorage", "CharStorage", "ByteStorage", "BoolStorage"}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that resolves only the globals a plain tensor checkpoint needs (stand-in for ``weights_only``)."""

    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED_GLOBALS or (module == "torch" and name in _STORAGE_NAMES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError("checkpoint refused: it needs the global %s.%s" % (module, name))


def _restricted_pickle_module() -> types.ModuleType:
    """Module-like object for ``torch.load(pickle_module=...)`` that enforces the allow-list on every load."""
    module = types.ModuleType("restricted_pickle")
    module.Unpickler = RestrictedUnpickler
    module.load = lambda handle, **kwargs: RestrictedUnpickler(handle, **kwargs).load()
    module.UnpicklingError = pickle.UnpicklingError
    return module


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint dict without executing code from the file."""
    import torch

    if "weights_only" in inspect.signature(torch.load).parameters:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    return torch.load(str(path), map_location="cpu", pickle_module=_restricted_pickle_module())


def load_config(path: str | Path) -> dict:
    """Read the released ``config.yaml``; its only non-plain YAML construct is ``!!python/tuple``."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    with open(str(path), "r") as handle:
        return yaml.load(handle, Loader=_Loader)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def build_semivl(semivl_dir: str | Path, checkpoint: str | Path, config_path: str | Path):
    """Build SemiVL from the released config and load ``best.pth`` (as ``third_party/unimatch/eval.py`` does).

    Changes the working directory to the SemiVL checkout (the model config names relative files, e.g. the text
    embeddings and ``pretrained/*.pth`` that ``init_weights`` reads).
    """
    import torch

    semivl_dir = Path(semivl_dir).resolve()
    for needed in ("model/builder.py", "pretrained/clip2mmseg_ViT16_clip_backbone.pth", "pretrained/resnet101_v1c-e67eebb6.pth"):
        if not (semivl_dir / needed).is_file():
            raise FileNotFoundError("%s is missing; run scripts/pseudo/setup_semivl_env.sh" % (semivl_dir / needed))
    os.chdir(str(semivl_dir))
    sys.path.insert(0, str(semivl_dir))
    from model.builder import build_model  # SemiVL's own module

    cfg = load_config(config_path)
    cfg.setdefault("text_embedding_variant", None)
    cfg.setdefault("pl_text", cfg["text_embedding_variant"])
    cfg["clip_encoder"] = None  # as eval.py: the MaskCLIP guidance branch is a training aid and not part of inference
    if cfg["dataset"] != "cityscapes" or cfg["nclass"] != NUM_CLASSES:
        raise ValueError("config.yaml is not a 19-class Cityscapes configuration")
    model = build_model(cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    state = load_checkpoint(checkpoint)
    weights = {}
    for key, value in state["model"].items():
        key = key[len("module."):] if key.startswith("module.") else key
        if "clip_encoder" not in key:
            weights[key] = value
    model.load_state_dict(weights)
    return model.eval(), cfg, int(state.get("epoch", -1))


# ----------------------------------------------------------------------------------------------
# Sliding window (same protocol as cora/pseudo/sliding_window.py)
# ----------------------------------------------------------------------------------------------
def sliding_window_probs(model_fn, image, num_classes: int, crop: int = DEFAULT_CROP, stride: int = DEFAULT_STRIDE):
    """Window-count-normalised softmax (C, H, W) of ``model_fn`` over a normalised (3, H, W) image."""
    import torch

    if not 1 <= stride <= crop:
        raise ValueError("need 1 <= stride <= crop, got stride=%d, crop=%d" % (stride, crop))
    _, height, width = image.shape
    total = torch.zeros(num_classes, height, width, dtype=torch.float32, device=image.device)
    counts = torch.zeros(height, width, dtype=torch.float32, device=image.device)
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            row1, col1 = min(height, row + crop), min(width, col + crop)
            logits = model_fn(image[None, :, row:row1, col:col1])
            total[:, row:row1, col:col1] += logits[0].float().softmax(dim=0)
            counts[row:row1, col:col1] += 1.0
    return total / counts


def preprocess(image: np.ndarray, device):
    """(H, W, 3) uint8 RGB -> normalised float (3, H, W) with ImageNet statistics (``ToTensor`` + ``Normalize``)."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return (tensor - mean) / std


# ----------------------------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------------------------
def prediction_paths(out_dir: str | Path, image_id: str) -> tuple:
    """(label path, confidence path); same id rules as ``cora/pseudo/io.py`` (normal-form relative path)."""
    posix = PurePosixPath(image_id)
    if (not image_id or posix.is_absolute() or ".." in posix.parts or "\\" in image_id
            or posix.as_posix() != image_id or not posix.name):
        raise ValueError("invalid image id %r" % image_id)
    base = Path(out_dir) / Path(image_id).parent
    stem = Path(image_id).name
    return base / (stem + LABEL_SUFFIX), base / (stem + CONF_SUFFIX)


def _write_png_atomic(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(buffer.getvalue())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, str(path))
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_prediction(out_dir: str | Path, image_id: str, label: np.ndarray, confidence: np.ndarray) -> None:
    """Same files as ``cora.pseudo.io.write_prediction``: confidence first, label last (it marks completion)."""
    label_path, conf_path = prediction_paths(out_dir, image_id)
    if label.dtype != np.uint8 or label.ndim != 2 or confidence.shape != label.shape:
        raise ValueError("label must be 2-D uint8 and match the confidence map")
    if not (np.isfinite(confidence).all() and confidence.min() >= 0.0 and confidence.max() <= 1.0):
        raise ValueError("confidence values must lie in [0, 1]")
    _write_png_atomic(conf_path, np.rint(confidence.astype(np.float32) * 255.0).astype(np.uint8))
    _write_png_atomic(label_path, label)


def load_rgb(root: str | Path, image_id: str) -> np.ndarray:
    from PIL import Image

    with Image.open(str(Path(root) / "leftImg8bit" / (image_id + "_leftImg8bit.png"))) as handle:
        return np.array(handle.convert("RGB"))


def load_train_ids(root: str | Path, image_id: str) -> np.ndarray:
    """Ground-truth trainIds from the labelIds file (255 = void)."""
    from PIL import Image

    lut = np.full(256, IGNORE, dtype=np.uint8)
    for label_id, train_id in _LABELID_TO_TRAINID.items():
        lut[label_id] = train_id
    with Image.open(str(Path(root) / "gtFine" / (image_id + "_gtFine_labelIds.png"))) as handle:
        return lut[np.array(handle)]


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    valid = (target < num_classes) & (pred < num_classes)
    index = target[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    return np.bincount(index, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    import torch

    if not 1 <= args.stride <= args.crop:
        raise ValueError("need 1 <= --stride <= --crop, got stride=%d, crop=%d" % (args.stride, args.crop))
    ids = [line.strip() for line in Path(args.id_file).read_text().splitlines() if line.strip()]
    ids = ids[args.shard_index::args.shard_count]
    config_path = Path(args.config) if args.config else Path(args.checkpoint).parent / "config.yaml"
    checkpoint = Path(args.checkpoint).resolve()
    config_path = config_path.resolve()
    out_dir = Path(args.out_dir).resolve()
    root = Path(args.cityscapes_root).resolve()
    result_json = Path(args.result_json).resolve()  # before build_semivl changes into the SemiVL checkout
    model, _, epoch = build_semivl(args.semivl_dir, checkpoint, config_path)
    device = torch.device(args.device)
    model = model.to(device)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    written = skipped = 0
    seconds = []
    with torch.no_grad():
        for position, image_id in enumerate(ids, 1):
            label_path, conf_path = prediction_paths(out_dir, image_id)
            if not args.evaluate and not args.overwrite and label_path.is_file() and conf_path.is_file():
                skipped += 1
                continue
            start = time.perf_counter()
            image = preprocess(load_rgb(root, image_id), device)
            probs = sliding_window_probs(model, image, NUM_CLASSES, args.crop, args.stride)
            confidence, label = probs.max(dim=0)
            label = label.to(torch.uint8).cpu().numpy()
            if args.evaluate:
                confusion += confusion_matrix(label, load_train_ids(root, image_id))
            else:
                write_prediction(out_dir, image_id, label, confidence.cpu().numpy())
                written += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
            if position % 25 == 0 or position == len(ids):
                sys.stdout.write("shard %d/%d: %d/%d images, %.2f s per image\n" % (
                    args.shard_index + 1, args.shard_count, position, len(ids), float(np.mean(seconds))))
                sys.stdout.flush()
    result = {
        "images": len(ids),
        "written": written,
        "skipped_existing": skipped,
        "seconds_per_image": seconds,
        "confusion": confusion.tolist() if args.evaluate else None,
        "extra": {"teacher_epoch": epoch, "checkpoint_sha256": sha256_file(checkpoint)},
    }
    result_json.write_text(json.dumps(result))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--semivl-dir", required=True, help="SemiVL checkout (commit 13ee65d), see scripts/pseudo/setup_semivl_env.sh")
    p.add_argument("--checkpoint", required=True, help="best.pth of a semivl_cs_<partition> release")
    p.add_argument("--config", default=None, help="default: config.yaml next to the checkpoint")
    p.add_argument("--cityscapes-root", required=True)
    p.add_argument("--id-file", required=True, help="text file with one image id <split>/<city>/<stem> per line")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--result-json", required=True, help="counts, timings and (with --evaluate) the confusion matrix")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop", type=int, default=DEFAULT_CROP)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--evaluate", action="store_true", help="accumulate a confusion matrix against the ground truth, write no labels")
    p.add_argument("--overwrite", action="store_true", help="recompute images that already have a prediction")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
