"""Token-level feature consistency loss (Eq. 8): unlabeled <SSEG> anchors against the labeled token bank."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

REDUCTIONS = ("sum", "mean")


def _sample(count: int, wanted: int, device: torch.device,
            generator: Optional[torch.Generator]) -> Optional[torch.Tensor]:
    """Indices of ``wanted`` of ``count`` entries without replacement; ``None`` selects all (0 or too few)."""
    if wanted == 0 or wanted >= count:
        return None
    gen_device = generator.device if generator is not None else torch.device("cpu")
    return torch.randperm(count, generator=generator, device=gen_device)[:wanted].to(device)


def token_contrastive_loss(anchors: torch.Tensor, anchor_classes: torch.Tensor, bank_features: torch.Tensor,
                           bank_classes: torch.Tensor, temperature: float, num_positives: int = 1,
                           num_negatives: int = 128, reduction: str = "sum",
                           generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, int]:
    """Eq. 8 with cosine similarity.

    For every anchor ``v`` with at least one same-class (``k+``) and one other-class (``k-``) bank entry,
    ``num_positives`` positives and ``num_negatives`` negatives are sampled without replacement (0 = all; if fewer
    exist, all of them). Each (anchor, positive) pair contributes
    ``-log(exp(s+/tau) / (exp(s+/tau) + sum_neg exp(s-/tau)))``. Anchors without positives or negatives, and
    anchors with a negative class id, are skipped. Bank entries are constants; gradients reach the anchors only.

    Args:
        anchors: ``(A, D)`` <SSEG> features of the unlabeled sequences.
        anchor_classes: ``(A,)`` class ids of the anchors.
        bank_features: ``(M, D)`` labeled token features (see :class:`cora.losses.token_bank.TokenBank`).
        bank_classes: ``(M,)`` class ids of the bank entries (entries with ids < 0 are never used).
        temperature: tau.
        reduction: ``"sum"`` (Eq. 8 as written) or ``"mean"`` over the (anchor, positive) pairs.
        generator: optional RNG for the sampling (CPU or the device of the generator).

    Returns:
        ``(loss, number of pairs)``. Without any pair the loss is ``anchors.sum() * 0``, which still depends on
        the anchors.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if num_positives < 0 or num_negatives < 0:
        raise ValueError(f"num_positives and num_negatives must be >= 0, got {num_positives}, {num_negatives}")
    if anchors.ndim != 2:
        raise ValueError(f"anchors must have shape (A, D), got {tuple(anchors.shape)}")
    if bank_features.ndim != 2 or (bank_features.shape[0] > 0 and bank_features.shape[1] != anchors.shape[1]):
        raise ValueError(f"bank_features must have shape (M, {anchors.shape[1]}), got {tuple(bank_features.shape)}")
    if anchor_classes.shape != (anchors.shape[0],) or bank_classes.shape != (bank_features.shape[0],):
        raise ValueError("anchor_classes / bank_classes must have one class id per anchor / bank entry")

    anchors = anchors.float()
    zero = anchors.sum() * 0.0
    if anchors.shape[0] == 0 or bank_features.shape[0] == 0:
        return zero, 0

    device = anchors.device
    bank_classes = bank_classes.to(device)
    similarity = F.normalize(anchors, dim=-1) @ F.normalize(bank_features.detach().to(device).float(), dim=-1).T
    similarity = similarity / temperature  # (A, M)

    terms = []
    for a, cls in enumerate(anchor_classes.tolist()):
        if cls < 0:
            continue
        positive_idx = (bank_classes == cls).nonzero(as_tuple=True)[0]
        negative_idx = ((bank_classes != cls) & (bank_classes >= 0)).nonzero(as_tuple=True)[0]
        if positive_idx.numel() == 0 or negative_idx.numel() == 0:
            continue
        pick = _sample(positive_idx.numel(), num_positives, device, generator)
        if pick is not None:
            positive_idx = positive_idx[pick]
        pick = _sample(negative_idx.numel(), num_negatives, device, generator)
        if pick is not None:
            negative_idx = negative_idx[pick]
        positive = similarity[a, positive_idx]
        negative_lse = torch.logsumexp(similarity[a, negative_idx], dim=0)
        # -log(e^p / (e^p + sum_neg e^n)) = logaddexp(p, lse(neg)) - p, one term per positive
        terms.append(torch.logaddexp(positive, negative_lse) - positive)

    if not terms:
        return zero, 0
    pairs = torch.cat(terms)
    loss = pairs.sum() if reduction == "sum" else pairs.mean()
    return loss, int(pairs.numel())
