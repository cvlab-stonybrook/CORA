"""Training objectives: labeled loss (Eq. 4-5), pixel weights (Eq. 6), unlabeled loss (Eq. 7-9).

The model output is used by duck typing (``mask_logits``, ``seg_embed``, ``seg_hidden``, ``ce_loss``), so this
module does not import ``cora.model``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from cora.config import LossConfig, TFCAConfig, UnlabeledConfig
from cora.data.batch import Batch
from cora.losses.consistency import confidence_weight, consistency_weight
from cora.losses.contrastive import token_contrastive_loss
from cora.losses.seg import dice_loss, sigmoid_ce_loss, weighted_seg_loss
from cora.losses.token_bank import TokenBank

TFCA_FEATURES = ("projected", "hidden")
WEIGHTINGS = ("consistency", "confidence", "none")


def _foreground(mask: torch.Tensor) -> torch.Tensor:
    """Target value 1 is foreground; 0, 255 (ignored) and anything else are background."""
    return (mask == 1).float()


def effective_dice_mode(ucfg: UnlabeledConfig) -> str:
    """Dice reading actually used by Eq. 7 for ``ucfg`` (supplement Sec. 2).

    With the confidence-threshold baseline (OCPL off, ``weighting="confidence"``) pixels below the threshold must
    contribute nothing to the BCE *and* the Dice term. The ``"plain"`` reading computes the Dice overlap over all
    pixels and only scales the scalar by the summed weights, so it would let those pixels supervise the Dice term.
    It is therefore replaced by ``"weighted"`` (lambda inside the Dice sums, still times the summed lambda).
    Every other combination keeps ``ucfg.dice_mode``.
    """
    if ucfg.weighting == "confidence" and ucfg.dice_mode == "plain":
        return "weighted"
    return ucfg.dice_mode


def _check_sequences(out, batch: Batch) -> None:
    if len(out.mask_logits) != batch.num_sequences or len(batch.masks) != batch.num_sequences:
        raise ValueError(
            f"{len(out.mask_logits)} mask logits and {len(batch.masks)} target masks for "
            f"{batch.num_sequences} sequences"
        )
    for i, (logits, mask) in enumerate(zip(out.mask_logits, batch.masks)):
        if logits.shape != mask.shape:
            raise ValueError(f"sequence {i}: logits {tuple(logits.shape)} and mask {tuple(mask.shape)} differ")


def labeled_loss(out, batch: Batch, lcfg: LossConfig) -> Dict[str, torch.Tensor]:
    """Eq. 4-5 with LISA's normalisation: per-mask BCE / Dice averaged over all masks of the batch.

    ``loss = ce_weight * ce + bce_weight * bce + dice_weight * dice``; the keys ``ce``, ``bce`` and ``dice`` hold
    the unweighted components.
    """
    _check_sequences(out, batch)
    n_total = batch.num_sequences
    if n_total == 0:
        raise ValueError("a labeled batch needs at least one sequence")
    bce = dice = 0.0
    if batch.seq_image.shape != (n_total,):
        raise ValueError(f"seq_image must have shape ({n_total},), got {tuple(batch.seq_image.shape)}")
    for image in batch.seq_image.unique().tolist():
        members = (batch.seq_image == image).nonzero(as_tuple=True)[0].tolist()
        pred = torch.stack([out.mask_logits[i].float() for i in members])
        gt = torch.stack([_foreground(batch.masks[i]).to(pred.device) for i in members])
        n_i = float(len(members))
        bce = bce + sigmoid_ce_loss(pred, gt, num_masks=n_i) * n_i
        dice = dice + dice_loss(pred, gt, num_masks=n_i) * n_i
    bce = bce / (n_total + 1e-8)
    dice = dice / (n_total + 1e-8)
    if out.ce_loss is None:
        if lcfg.ce_weight != 0:
            raise ValueError("the model output has no ce_loss but loss.ce_weight is not zero")
        ce = bce * 0.0
    else:
        ce = out.ce_loss
    loss = lcfg.ce_weight * ce + lcfg.bce_weight * bce + lcfg.dice_weight * dice
    return {"loss": loss, "ce": ce, "bce": bce, "dice": dice}


def pixel_weights(batch: Batch, teacher_logits: Optional[List[torch.Tensor]],
                  ucfg: UnlabeledConfig) -> List[torch.Tensor]:
    """One ``(H, W)`` weight map lambda per trained sequence of an unlabeled batch (Sec. 3.2.1).

    ``"consistency"``: Eq. 6 over the teacher logits of the K-1 rephrasings of the sequence (grouped with
    ``batch.aux.group``); ``"confidence"``: threshold on ``batch.confidence``; ``"none"``: ones.
    """
    n = batch.num_sequences
    if ucfg.weighting not in WEIGHTINGS:
        raise ValueError(f"unlabeled.weighting must be one of {WEIGHTINGS}, got {ucfg.weighting!r}")
    if ucfg.weighting == "none":
        return [torch.ones(m.shape, dtype=torch.float32, device=m.device) for m in batch.masks]
    if ucfg.weighting == "confidence":
        if batch.confidence is None or len(batch.confidence) != n or any(c is None for c in batch.confidence):
            raise ValueError("unlabeled.weighting='confidence' needs a confidence map for every sequence")
        return [confidence_weight(c, ucfg.confidence_threshold).to(m.device)
                for c, m in zip(batch.confidence, batch.masks)]

    if teacher_logits is None or batch.aux is None or batch.aux.group is None:
        raise ValueError("unlabeled.weighting='consistency' needs teacher logits and an aux batch with groups")
    group = batch.aux.group
    if len(teacher_logits) != group.shape[0]:
        raise ValueError(f"{len(teacher_logits)} teacher logits for {group.shape[0]} aux sequences")
    if group.numel() > 0 and (int(group.min()) < 0 or int(group.max()) >= n):
        raise ValueError(f"batch.aux.group must lie in [0, {n}), got range [{int(group.min())}, {int(group.max())}]")
    weights = []
    for i in range(n):
        members = (group == i).nonzero(as_tuple=True)[0].tolist()
        if not members:
            raise ValueError(f"sequence {i} has no aux rephrasings (unlabeled.num_queries must be >= 2)")
        stacked = [teacher_logits[j] for j in members]
        if any(t.shape != batch.masks[i].shape for t in stacked):
            raise ValueError(f"sequence {i}: teacher logits do not match the mask shape {tuple(batch.masks[i].shape)}")
        weights.append(consistency_weight(torch.stack(stacked), ucfg.weight_mode, ucfg.variance_input,
                                          ucfg.exp_scale))
    return weights


def unlabeled_loss(out, batch: Batch, weights: List[torch.Tensor], bank: Optional[TokenBank],
                   ucfg: UnlabeledConfig, tcfg: TFCAConfig) -> Dict[str, torch.Tensor]:
    """Eq. 7 summed over the trained sequences plus Eq. 8, both multiplied by sigma (Eq. 9).

    Returns ``loss`` (scaled by ``ucfg.weight``), the unscaled ``seg`` (Eq. 7 with the BCE / Dice weights of
    ``ucfg``), its parts ``bce`` and ``dice``, ``tfca`` (Eq. 8), ``weight_mean`` and ``pairs`` for logging.
    With ``weighting="confidence"`` the default ``dice_mode="plain"`` is read as ``"weighted"`` (see
    :func:`effective_dice_mode`) so that below-threshold pixels leave the Dice term too.
    ``bank`` may be ``None`` only when ``tcfg.enabled`` is False; otherwise a ``ValueError`` is raised.
    """
    _check_sequences(out, batch)
    n = batch.num_sequences
    if n == 0:
        raise ValueError("an unlabeled batch needs at least one sequence")
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weight maps for {n} sequences")
    dice_mode = effective_dice_mode(ucfg)
    bce = dice = 0.0
    for logits, mask, weight in zip(out.mask_logits, batch.masks, weights):
        if weight.shape != logits.shape:
            raise ValueError(f"weight map {tuple(weight.shape)} does not match logits {tuple(logits.shape)}")
        b, d = weighted_seg_loss(logits, _foreground(mask).to(logits.device), weight.to(logits.device),
                                 ucfg.reduction, dice_mode)
        bce = bce + b
        dice = dice + d
    seg = ucfg.bce_weight * bce + ucfg.dice_weight * dice

    if tcfg.enabled:
        if bank is None:
            raise ValueError("tfca.enabled is True but no TokenBank was given; pass a bank or disable tfca")
        if tcfg.feature not in TFCA_FEATURES:
            raise ValueError(f"tfca.feature must be one of {TFCA_FEATURES}, got {tcfg.feature!r}")
        anchors = out.seg_embed if tcfg.feature == "projected" else out.seg_hidden
        bank_features, bank_classes = bank.get()
        tfca, pairs = token_contrastive_loss(
            anchors, batch.class_ids, bank_features, bank_classes, tcfg.temperature, tcfg.num_positives,
            tcfg.num_negatives, tcfg.reduction)
    else:
        tfca, pairs = seg * 0.0, 0

    weight_mean = torch.stack([w.float().mean() for w in weights]).mean().detach()
    return {
        "loss": ucfg.weight * (seg + tfca),
        "seg": seg,
        "bce": bce,
        "dice": dice,
        "tfca": tfca,
        "weight_mean": weight_mean,
        "pairs": torch.tensor(float(pairs), device=seg.device),
    }
