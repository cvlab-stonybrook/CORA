"""Image and anchor preprocessing (paper Sec. 3.1, Sec. 3.3 and supplement Sec. 5).

Two views of every image go into CORA, as in LISA: the SAM view (longest side resized to 1024, zero padded
bottom/right) feeds the mask decoder, the CLIP view feeds LLaVA. Conditional instructions (Sec. 3.3) add an
anchor view built as in SegLLM: the masked, tightly cropped, square-padded anchor pixels for the anchor mask
encoder and the anchor box in the SAM frame for the anchor box encoder.

All functions take RGB ``uint8`` arrays of shape ``(H, W, 3)`` and return float32 tensors.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

# segment-anything's Sam.pixel_mean / pixel_std, applied to 0-255 values (LISA utils/dataset.py).
SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)
# OpenAI CLIP / LLaVA-1.5 CLIPImageProcessor statistics.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_MODES = ("pad", "center_crop", "squash")


def _check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        shape = getattr(image, "shape", None)
        dtype = getattr(image, "dtype", type(image).__name__)
        raise ValueError(f"expected an RGB uint8 array of shape (H, W, 3), got {dtype} with shape {shape}")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"image is empty: shape {image.shape}")


def _resize(image: np.ndarray, height: int, width: int, resample: int) -> np.ndarray:
    """PIL resize of a uint8 RGB array to ``(height, width)``."""
    if image.shape[:2] == (height, width):
        return image
    return np.asarray(Image.fromarray(image).resize((width, height), resample))


def sam_resize_shape(height: int, width: int, size: int = 1024) -> Tuple[int, int]:
    """(h, w) after resizing the longest side to ``size``: segment_anything ``get_preprocess_shape``."""
    scale = size * 1.0 / max(height, width)
    return int(height * scale + 0.5), int(width * scale + 0.5)


def sam_preprocess(image: np.ndarray, size: int = 1024) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """SAM view of an image, exactly as LISA prepares it (Sec. 3.1).

    The longest side is resized to ``size`` with PIL bilinear interpolation (what
    ``ResizeLongestSide.apply_image`` does; ``apply_image_torch`` is not used because it is broken in
    segment-anything 1.0), the 0-255 values are normalised with SAM's mean / std, and the result is padded with
    zeros on the bottom and right to ``(3, size, size)``. Returns the tensor and the ``(h, w)`` it had before
    padding.
    """
    _check_image(image)
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    new_h, new_w = sam_resize_shape(image.shape[0], image.shape[1], size)
    resized = _resize(image, new_h, new_w, Image.BILINEAR)
    mean = torch.tensor(SAM_PIXEL_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(SAM_PIXEL_STD, dtype=torch.float32).view(3, 1, 1)
    x = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float()
    x = (x - mean) / std
    out = torch.zeros(3, size, size, dtype=torch.float32)
    out[:, :new_h, :new_w] = x
    return out, (new_h, new_w)


def _expand_to_square(image: np.ndarray, colour: Tuple[int, int, int]) -> np.ndarray:
    """LLaVA-1.5 ``expand2square``: centre the image on a square canvas filled with ``colour``."""
    h, w = image.shape[:2]
    if h == w:
        return image
    side = max(h, w)
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[...] = np.asarray(colour, dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top : top + h, left : left + w] = image
    return canvas


def clip_preprocess(image: np.ndarray, size: int, mode: str = "pad") -> torch.Tensor:
    """CLIP view of an image: ``(3, size, size)`` float32 normalised with the LLaVA-1.5 CLIP statistics.

    Modes (``DataConfig.clip_preprocess``), all with bicubic resampling as the CLIP processor:

    * ``"pad"``: centre the image on a square canvas filled with the CLIP mean colour (LLaVA-1.5's
      ``image_aspect_ratio="pad"``), then resize; keeps the whole frame;
    * ``"center_crop"``: resize the shortest side to ``size`` and crop the centre (the plain CLIP processor,
      LISA's behaviour); drops the sides of a wide frame;
    * ``"squash"``: resize to ``size x size`` ignoring the aspect ratio.
    """
    _check_image(image)
    if mode not in CLIP_MODES:
        raise ValueError(f"unknown CLIP preprocessing mode {mode!r}; expected one of {CLIP_MODES}")
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    h, w = image.shape[:2]
    if mode == "pad":
        square = _expand_to_square(image, tuple(int(c * 255) for c in CLIP_MEAN))
        view = _resize(square, size, size, Image.BICUBIC)
    elif mode == "squash":
        view = _resize(image, size, size, Image.BICUBIC)
    else:
        short = min(h, w)
        new_h, new_w = (size, int(size * w / short)) if h <= w else (int(size * h / short), size)
        resized = _resize(image, new_h, new_w, Image.BICUBIC)
        top, left = (new_h - size) // 2, (new_w - size) // 2
        view = resized[top : top + size, left : left + size]
    x = torch.from_numpy(np.array(view)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std


def _mask_extent(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Inclusive (row_min, col_min, row_max, col_max) of the non-zero pixels of a 2-D mask."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ValueError("the anchor mask is empty")
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def anchor_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked anchor pixels for the anchor mask encoder (SegLLM ``get_bitmask_bbox_encode``; supplement Sec. 5).

    Pixels outside the mask are set to zero, the tight bounding box of the mask is cut out, and the crop is
    zero padded to a square at its bottom / right (top-left aligned). Returns ``(S, S, 3)`` uint8 with
    ``S = max(box height, box width)``. Non-zero mask values count as inside.
    """
    _check_image(image)
    mask = np.asarray(mask)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} differs from image shape {image.shape[:2]}")
    inside = mask != 0
    r0, c0, r1, c1 = _mask_extent(inside)
    crop = np.where(inside[r0 : r1 + 1, c0 : c1 + 1, None], image[r0 : r1 + 1, c0 : c1 + 1], 0).astype(np.uint8)
    h, w = crop.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=np.uint8)
    out[:h, :w] = crop
    return out


def anchor_box(mask: np.ndarray, size: int = 1024) -> torch.Tensor:
    """Anchor box for the anchor box encoder: ``[y0, x0, y1, x1] / size`` in the SAM frame (supplement Sec. 5).

    As in SegLLM the coordinates are inclusive pixel indices of the mask extent after the SAM resize (longest
    side to ``size``, padding at the bottom / right) and are divided by ``size``, not by the image size.
    The extent is mapped analytically (each source pixel ``i`` covers ``[i * s, (i + 1) * s)`` in the resized
    frame, ``s = size / max(H, W)``), so a thin mask never vanishes in the resize. SegLLM instead resizes the
    mask and takes its non-zero extent; the two differ by at most 2 px of ``size`` (0.002 after division).
    """
    mask = np.asarray(mask)
    r0, c0, r1, c1 = _mask_extent(mask != 0)
    new_h, new_w = sam_resize_shape(mask.shape[0], mask.shape[1], size)
    scale = size / max(mask.shape)

    def project(lo: int, hi: int, limit: int) -> Tuple[int, int]:
        start = min(int(np.floor(lo * scale)), limit - 1)
        end = min(max(int(np.ceil((hi + 1) * scale)) - 1, start), limit - 1)
        return start, end

    y0, y1 = project(r0, r1, new_h)
    x0, x1 = project(c0, c1, new_w)
    return torch.tensor([y0, x0, y1, x1], dtype=torch.float32) / float(size)
