"""Sliding-window inference of the UniMatch / SemiVL Cityscapes protocol (paper Sec. 4.1).

The teachers are evaluated on the full 1024 x 2048 image with square windows of ``crop`` = 801 pixels and
a stride of ``int(801 * 2 / 3)`` = 534 pixels. Windows start at 0, stride, 2 * stride, ... while the start
is smaller than the image size; windows that reach over the border are cut (not padded), so the network
sees inputs smaller than 801 at the right and bottom edge. The per-window softmax is summed into a
buffer (``supervised.py:evaluate`` of UniMatch). Here the sum is divided by the number of windows that
cover each pixel; the argmax (the label) is unchanged by that division, and the maximum becomes a proper
probability that is used as the confidence map of the pseudo-label (CORA needs it for the sigma >= 0.9
threshold of the supplement, Sec. 2).
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from torch import Tensor

DEFAULT_CROP = 801
DEFAULT_STRIDE = 534  # int(801 * 2 / 3)


def window_starts(size: int, stride: int = DEFAULT_STRIDE) -> List[int]:
    """Start offsets along one axis: 0, stride, ... while the start is below ``size``."""
    if size < 1 or stride < 1:
        raise ValueError(f"size and stride must be positive, got size={size}, stride={stride}")
    return list(range(0, size, stride))


def window_boxes(height: int, width: int, crop: int = DEFAULT_CROP, stride: int = DEFAULT_STRIDE) -> List[Tuple[int, int, int, int]]:
    """All windows as (row0, row1, col0, col1), row-major, clipped at the image border."""
    if crop < 1:
        raise ValueError(f"crop must be positive, got {crop}")
    if stride > crop:
        raise ValueError(f"stride {stride} exceeds crop {crop}: pixels between the windows would be covered by none")
    return [
        (row, min(height, row + crop), col, min(width, col + crop))
        for row in window_starts(height, stride)
        for col in window_starts(width, stride)
    ]


def window_counts(height: int, width: int, crop: int = DEFAULT_CROP, stride: int = DEFAULT_STRIDE) -> Tensor:
    """(H, W) number of windows covering each pixel."""
    counts = torch.zeros(height, width)
    for row0, row1, col0, col1 in window_boxes(height, width, crop, stride):
        counts[row0:row1, col0:col1] += 1.0
    return counts


@torch.no_grad()
def sliding_window_probs(
    model_fn: Callable[[Tensor], Tensor],
    image: Tensor,
    num_classes: int,
    crop: int = DEFAULT_CROP,
    stride: int = DEFAULT_STRIDE,
) -> Tensor:
    """Window-count-normalised softmax of a segmentation network over a full image.

    Args:
        model_fn: maps a normalised window (1, 3, h, w) to logits (1, num_classes, h, w).
        image: normalised image (3, H, W), already on the device the network uses.
        num_classes: number of classes C.
    Returns:
        (C, H, W) float32 probabilities (they sum to 1 over C at every pixel); ``probs.max(0)`` gives the
        confidence map and its argmax the label map.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image must have shape (3, H, W), got {tuple(image.shape)}")
    _, height, width = image.shape
    total = torch.zeros(num_classes, height, width, dtype=torch.float32, device=image.device)
    counts = torch.zeros(height, width, dtype=torch.float32, device=image.device)
    for row0, row1, col0, col1 in window_boxes(height, width, crop, stride):
        logits = model_fn(image[None, :, row0:row1, col0:col1])
        expected = (1, num_classes, row1 - row0, col1 - col0)
        if tuple(logits.shape) != expected:
            raise ValueError(f"model_fn returned logits of shape {tuple(logits.shape)}, expected {expected}")
        total[:, row0:row1, col0:col1] += logits[0].float().softmax(dim=0)
        counts[row0:row1, col0:col1] += 1.0
    return total / counts
