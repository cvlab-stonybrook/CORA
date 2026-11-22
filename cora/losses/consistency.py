"""Per-pixel weights of the unlabeled loss: output consistency (Eq. 6) and the confidence-threshold baseline."""
from __future__ import annotations

import torch

WEIGHT_MODES = ("stable_high", "variance", "one_minus", "exp")
VARIANCE_INPUTS = ("prob", "binary")
MAX_BINARY_VARIANCE = 0.25  # variance of a {0, 1}-valued variable is at most 1/4


def pixel_variance(predictions: torch.Tensor) -> torch.Tensor:
    """Eq. 6: population variance (divisor ``K - 1``, the number of predictions) over dim 0 of ``(K-1, H, W)``."""
    if predictions.ndim != 3 or predictions.shape[0] < 1:
        raise ValueError(f"predictions must have shape (K-1, H, W) with K-1 >= 1, got {tuple(predictions.shape)}")
    return predictions.var(dim=0, unbiased=False)


def consistency_weight(mask_logits: torch.Tensor, mode: str, variance_input: str = "prob",
                       exp_scale: float = 10.0) -> torch.Tensor:
    """Turn the ``K-1`` mask logits of the rephrased queries into the pixel weights lambda (Eq. 6).

    Args:
        mask_logits: ``(K-1, H, W)`` logits at the original image resolution.
        mode: how the variance ``v`` becomes lambda: ``"stable_high"`` = clamp(1 - v / 0.25, 0, 1) (stable pixels
            are emphasised, as the text of Sec. 3.2.1 says), ``"variance"`` = v (Eq. 6 as printed),
            ``"one_minus"`` = 1 - v, ``"exp"`` = exp(-exp_scale * v).
        variance_input: ``"prob"`` takes the variance of sigmoid probabilities, ``"binary"`` of the masks binarised
            at logit > 0.
        exp_scale: slope of the ``"exp"`` mode.

    Returns:
        ``(H, W)`` float32 weights without gradient.
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"weight mode must be one of {WEIGHT_MODES}, got {mode!r}")
    if variance_input not in VARIANCE_INPUTS:
        raise ValueError(f"variance_input must be one of {VARIANCE_INPUTS}, got {variance_input!r}")
    if mode == "exp" and exp_scale < 0:
        raise ValueError(f"exp_scale must be non-negative, got {exp_scale}")
    with torch.no_grad():
        logits = mask_logits.detach().float()
        values = torch.sigmoid(logits) if variance_input == "prob" else (logits > 0).float()
        variance = pixel_variance(values)
        if mode == "stable_high":
            return (1.0 - variance / MAX_BINARY_VARIANCE).clamp(0.0, 1.0)
        if mode == "variance":
            return variance
        if mode == "one_minus":
            return 1.0 - variance
        return torch.exp(-exp_scale * variance)


def confidence_weight(confidence: torch.Tensor, threshold: float) -> torch.Tensor:
    """Baseline of the ablation (supplement Sec. 2): 1.0 where the teacher confidence >= ``threshold`` else 0.0."""
    return (confidence.detach() >= threshold).float()
