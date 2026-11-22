"""Segmentation losses: LISA's BCE / Dice (Eq. 4) and the pixel-weighted loss of one unlabeled mask (Eq. 7)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

REDUCTIONS = ("sum", "mean")
DICE_MODES = ("plain", "weighted", "weighted_unscaled", "off")


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_masks: float, scale: float = 1000.0,
              eps: float = 1e-6) -> torch.Tensor:
    """LISA's Dice loss (Eq. 4), summed over the ``N`` masks of ``(N, H, W)`` inputs and divided by ``num_masks``.

    Adapted from LISA (``model/LISA.py::dice_loss``, Apache-2.0): ``scale`` divides numerator and denominator.
    """
    inputs = logits.sigmoid().flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)


def sigmoid_ce_loss(logits: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    """LISA's binary cross-entropy (Eq. 4): per-mask pixel mean, summed over masks, divided by ``num_masks``.

    Adapted from LISA (``model/LISA.py::sigmoid_ce_loss``, Apache-2.0).
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    return values.sum() if reduction == "sum" else values.mean()


def weighted_seg_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, reduction: str,
                      dice_mode: str, scale: float = 1000.0, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 7 for ONE mask: ``sum_i lambda_i (BCE_i + DICE)`` split into its BCE and Dice parts.

    Args:
        logits: ``(H, W)`` prediction logits of the trained branch (carry the gradient).
        target: ``(H, W)`` pseudo-mask in ``{0, 1}``.
        weight: ``(H, W)`` per-pixel weights lambda; treated as a constant (detached).
        reduction: ``"sum"`` (Eq. 7 as written) or ``"mean"`` over the ``H * W`` pixels.
        dice_mode: Dice has no per-pixel value, so Eq. 7 needs a reading (Sec. 3.2.1):
            ``"plain"``: ``D`` = soft Dice of the two masks, multiplied by ``reduce_i(lambda_i)``;
            ``"weighted"``: ``D`` = soft Dice with lambda inside its sums, times ``reduce_i(lambda_i)``;
            ``"weighted_unscaled"``: the lambda-weighted soft Dice alone;
            ``"off"``: the Dice term is zero.
        scale, eps: LISA's Dice constants.

    Returns:
        ``(bce, dice)`` with ``bce = reduce_i(lambda_i * BCE_i)``. With ``lambda == 1`` and reduction ``"mean"``
        both equal LISA's per-mask values for every mode except ``"off"``.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    if dice_mode not in DICE_MODES:
        raise ValueError(f"dice_mode must be one of {DICE_MODES}, got {dice_mode!r}")
    if logits.ndim != 2 or logits.shape != target.shape or logits.shape != weight.shape:
        raise ValueError(
            f"logits, target and weight must share one (H, W) shape, got {tuple(logits.shape)}, "
            f"{tuple(target.shape)}, {tuple(weight.shape)}"
        )
    logits = logits.float()
    target = target.float()
    weight = weight.detach().float()

    bce = _reduce(weight * F.binary_cross_entropy_with_logits(logits, target, reduction="none"), reduction)
    if dice_mode == "off":
        return bce, logits.sum() * 0.0

    prob = logits.sigmoid() / scale
    scaled_target = target / scale
    if dice_mode == "plain":
        numerator = 2 * (prob * target).sum()
        denominator = prob.sum() + scaled_target.sum()
    else:
        numerator = 2 * (weight * prob * target).sum()
        denominator = (weight * prob).sum() + (weight * scaled_target).sum()
    dice = 1 - (numerator + eps) / (denominator + eps)
    if dice_mode != "weighted_unscaled":
        dice = dice * _reduce(weight, reduction)
    return bce, dice
