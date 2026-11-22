"""UniMatch teacher (paper Sec. 4.1): DeepLabV3+ ResNet-101 of LiheYoung/UniMatch run as a pseudo-labeller.

The network definition is imported from a UniMatch checkout (``third_party/UniMatch``); nothing of it is
copied or edited. Its constructor loads the ImageNet ResNet-101 file from the working-directory-relative
path ``pretrained/resnet101.pth``; :func:`build_unimatch` provides that path through a temporary folder and
restores the caller's working directory afterwards.

Checkpoints are third-party pickles. They are read with ``torch.load(..., weights_only=True)`` and exactly the
three numpy types a UniMatch checkpoint stores (the numpy float ``previous_best``) are allow-listed;
anything else is refused with the name of the offending type.

Inference follows the repository's Cityscapes evaluation: full-resolution image, ImageNet mean / std,
crop-801 sliding window with stride 534 (:mod:`cora.pseudo.sliding_window`), batch size 1, no flip or
multi-scale test time augmentation.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from cora.pseudo.sliding_window import DEFAULT_CROP, DEFAULT_STRIDE, sliding_window_probs

try:  # numpy >= 2 moved the module; UniMatch checkpoints reference the old name "numpy.core.multiarray.scalar"
    from numpy._core import multiarray as _multiarray
except ImportError:  # pragma: no cover - numpy 1.x
    from numpy.core import multiarray as _multiarray  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# configs/cityscapes.yaml of UniMatch (ResNet-101, output stride 16, ASPP dilations 6 / 12 / 18).
CITYSCAPES_MODEL_CONFIG = {
    "nclass": 19,
    "backbone": "resnet101",
    "replace_stride_with_dilation": [False, False, True],
    "dilations": [6, 12, 18],
}

MODEL_CONFIGS = {"cityscapes": CITYSCAPES_MODEL_CONFIG}
# (crop, stride) of the sliding window: Cityscapes protocol (801, 534).
DATASET_WINDOWS = {"cityscapes": (DEFAULT_CROP, DEFAULT_STRIDE)}

# Types stored in a UniMatch checkpoint besides tensors, dicts, lists and ints: the numpy float
# ``previous_best``. The scalar constructor is listed under the name numpy 1.x wrote into the released files
# and under its current name, so that files saved by either numpy generation load.
NUMPY_SAFE_GLOBALS = [
    (_multiarray.scalar, "numpy.core.multiarray.scalar"),
    _multiarray.scalar,
    np.dtype,
    np.dtypes.Float64DType,
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file (recorded in the manifest of every pseudo-label folder)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    """Read a UniMatch checkpoint with ``weights_only=True`` and the numpy allow-list.

    Returns the checkpoint dict (keys ``model``, ``optimizer``, ``epoch``, ``previous_best``).
    Raises RuntimeError naming the refused type when the file needs anything else.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"UniMatch checkpoint not found: {path}")
    try:
        with torch.serialization.safe_globals(NUMPY_SAFE_GLOBALS):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        reason = str(error).split("WeightsUnpickler error:")[-1].strip().splitlines()[0]
        raise RuntimeError(f"{path} cannot be loaded with weights_only=True: {reason}") from error
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path} is not a UniMatch checkpoint (no 'model' entry)")
    return checkpoint


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove the ``module.`` prefix that DistributedDataParallel adds to every key."""
    return {key[len("module."):] if key.startswith("module.") else key: value for key, value in state_dict.items()}


@contextlib.contextmanager
def _backbone_in_working_directory(backbone_weights: Path) -> Iterator[None]:
    """Make ``pretrained/resnet101.pth`` resolvable from the working directory, then restore it."""
    backbone_weights = Path(backbone_weights).resolve()
    if not backbone_weights.is_file():
        raise FileNotFoundError(f"ResNet-101 backbone file not found: {backbone_weights}")
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="unimatch_cwd_") as folder:
        (Path(folder) / "pretrained").mkdir()
        (Path(folder) / "pretrained" / "resnet101.pth").symlink_to(backbone_weights)
        os.chdir(folder)
        try:
            yield
        finally:
            os.chdir(previous)


def _import_deeplab(checkout: Path):
    """Import ``model.semseg.deeplabv3plus`` from a UniMatch checkout and verify where it came from."""
    checkout = Path(checkout).resolve()
    module_file = checkout / "model" / "semseg" / "deeplabv3plus.py"
    if not module_file.is_file():
        raise FileNotFoundError(f"{checkout} is not a UniMatch checkout (missing model/semseg/deeplabv3plus.py)")
    loaded = sys.modules.get("model.semseg.deeplabv3plus")
    if loaded is None:
        sys.path.insert(0, str(checkout))
        try:
            loaded = importlib.import_module("model.semseg.deeplabv3plus")
        finally:
            sys.path.remove(str(checkout))
    if Path(loaded.__file__).resolve() != module_file:
        raise ImportError(
            f"a top-level package named 'model' from {loaded.__file__} is already imported; "
            f"cannot import the UniMatch model from {checkout}"
        )
    return loaded


def model_config(dataset: str) -> dict:
    """Network configuration of a dataset (``cityscapes``: 19 classes)."""
    if dataset not in MODEL_CONFIGS:
        raise ValueError(f"no UniMatch model configuration for dataset {dataset!r}; expected one of {sorted(MODEL_CONFIGS)}")
    return dict(MODEL_CONFIGS[dataset])


def build_unimatch(checkout: Path, backbone_weights: Path, config: Optional[dict] = None) -> nn.Module:
    """Build UniMatch's DeepLabV3+ (random head, ImageNet backbone) from a checkout.

    ``config`` defaults to :data:`CITYSCAPES_MODEL_CONFIG`. The caller's working directory is unchanged.
    """
    module = _import_deeplab(checkout)
    with _backbone_in_working_directory(backbone_weights):
        return module.DeepLabV3Plus(dict(config or CITYSCAPES_MODEL_CONFIG))


def preprocess_image(image: np.ndarray, device: torch.device) -> Tensor:
    """(H, W, 3) uint8 RGB -> normalised float32 (3, H, W) (``ToTensor`` + ``Normalize`` with ImageNet statistics)."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"image must be (H, W, 3) uint8 RGB, got {image.dtype} with shape {image.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return (tensor - mean) / std


class UniMatchTeacher:
    """A loaded UniMatch network that turns images into (label, confidence) pseudo-labels (Sec. 4.1)."""

    def __init__(
        self,
        checkout: Path,
        checkpoint: Path,
        backbone_weights: Path,
        device: str = "cuda",
        crop: Optional[int] = None,
        stride: Optional[int] = None,
        dataset: str = "cityscapes",
    ) -> None:
        config = model_config(dataset)
        default_crop, default_stride = DATASET_WINDOWS[dataset]
        self.dataset = dataset
        self.device = torch.device(device)
        self.crop = default_crop if crop is None else crop
        self.stride = default_stride if stride is None else stride
        self.num_classes = config["nclass"]
        self.checkpoint_path = Path(checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        model = build_unimatch(checkout, backbone_weights, config)
        state = load_checkpoint(self.checkpoint_path)
        model.load_state_dict(strip_module_prefix(state["model"]), strict=True)
        self.epoch = int(state["epoch"])
        self.reported_miou = float(state["previous_best"])
        self.model = model.to(self.device).eval()
        logger.info("UniMatch checkpoint %s: epoch %d, best val mIoU stored in it %.2f", self.checkpoint_path.name,
                    self.epoch, self.reported_miou)

    @torch.no_grad()
    def probabilities(self, image: np.ndarray) -> Tensor:
        """(num_classes, H, W) window-count-normalised softmax of an RGB uint8 image, on the model's device."""
        tensor = preprocess_image(image, self.device)
        return sliding_window_probs(self.model, tensor, self.num_classes, self.crop, self.stride)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(label uint8 (H, W), confidence float32 (H, W)) of an RGB uint8 image (confidence = max probability)."""
        confidence, label = self.probabilities(image).max(dim=0)
        label = label.to(torch.uint8).cpu().numpy()
        return label, confidence.cpu().numpy()
