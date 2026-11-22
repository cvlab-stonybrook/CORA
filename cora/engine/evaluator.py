"""Evaluation with cumulative IoU (cIoU), the paper's metric (Sec. 4.1), computed as in LISA.

cIoU  = sum of intersections / sum of unions over all (image, query) pairs.
gIoU  = mean of the per-pair IoU (a pair with an empty target and an empty prediction counts as 1).
Predictions are mask logits thresholded at 0 at the original image resolution; target pixels equal to
255 are ignored.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from cora import constants as C
from cora.config import Config
from cora.engine import dist as D


def intersection_union(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """Foreground intersection and union of a binary prediction and a target with values 0 / 1 / 255."""
    if pred.shape != target.shape:
        raise ValueError(f"prediction {tuple(pred.shape)} and target {tuple(target.shape)} differ in shape")
    valid = target != C.IGNORE_LABEL
    p = pred.bool() & valid
    t = (target == 1) & valid
    return float((p & t).sum().item()), float((p | t).sum().item())


class IoUMeter:
    """Per group: [sum of intersections, sum of unions, sum of per-pair IoU, number of pairs, pairs without a mask]."""

    def __init__(self) -> None:
        self.stats: Dict[str, List[float]] = {}

    def update(self, group: str, inter: float, union: float, no_mask: bool = False) -> None:
        s = self.stats.setdefault(group, [0.0] * 5)
        s[0] += inter
        s[1] += union
        s[2] += inter / (union + 1e-5) + (1.0 if union == 0 else 0.0)
        s[3] += 1.0
        s[4] += 1.0 if no_mask else 0.0

    def merge(self, stats: Dict[str, List[float]]) -> None:
        for group, values in stats.items():
            s = self.stats.setdefault(group, [0.0] * 5)
            for i in range(5):
                s[i] += values[i]

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Percentages, as reported in the paper's tables."""
        return {
            group: {
                "ciou": 100.0 * s[0] / (s[1] + 1e-10),
                "giou": 100.0 * s[2] / max(s[3], 1.0),
                "n": int(s[3]),
                "no_mask": int(s[4]),  # generate mode: answers in which the model wrote no <SSEG>
            }
            for group, s in sorted(self.stats.items())
        }


def _groups(batch, index: int) -> List[str]:
    groups = ["all", batch.set_names[index]]
    if batch.is_long_query and batch.is_long_query[index] is not None:
        groups.append("long" if batch.is_long_query[index] else "short")
    return groups


@torch.no_grad()
def evaluate(model, loader: Iterable, device: torch.device, mode: str = "teacher_forced",
             max_new_tokens: int = 32, max_batches: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Evaluate ``model`` on every batch of ``loader``; returns metrics per group.

    Groups are ``all``, each instruction set name, and ``long`` / ``short`` when the batch carries the
    ReasonSeg query-length flag. ``mode="teacher_forced"`` feeds the answer with ``<SSEG>`` as LISA's
    validation does; ``mode="generate"`` lets the model write the answer, and a sequence in which it emits
    no ``<SSEG>`` counts as an empty prediction. In a distributed run every rank evaluates its own shard
    and the statistics are summed.
    """
    if mode not in ("teacher_forced", "generate"):
        raise ValueError(f"unknown evaluation mode {mode!r}")
    was_training = model.training
    model.eval()
    meter = IoUMeter()
    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch.to(device)
            if mode == "generate":
                _, masks = model.generate(batch, max_new_tokens=max_new_tokens)
            else:
                masks = model(batch, compute_text_loss=False).mask_logits
            for j, logits in enumerate(masks):
                target = batch.masks[j]
                pred = torch.zeros_like(target, dtype=torch.bool) if logits is None else logits > 0
                inter, union = intersection_union(pred, target)
                for group in _groups(batch, j):
                    meter.update(group, inter, union, no_mask=logits is None)
    finally:
        model.train(was_training)
    merged = IoUMeter()
    for stats in D.gather_objects(meter.stats):
        merged.merge(stats)
    return merged.compute()


def shard_indices(length: int, rank: int, world_size: int, max_samples: Optional[int] = None) -> List[int]:
    """Indices of a dataset that this rank evaluates.

    With ``max_samples`` the subset is spread evenly over the whole dataset (so that every instruction set
    is represented) before it is split between the ranks.
    """
    indices = list(range(length))
    if max_samples is not None and 0 < max_samples < length:
        indices = [int(i * length / max_samples) for i in range(max_samples)]
    return indices[rank::world_size]


def format_results(results: Dict[str, Dict[str, float]]) -> str:
    width = max([14] + [len(group) + 2 for group in results])
    show_no_mask = any(values.get("no_mask", 0) for values in results.values())
    header = f"{'group':<{width}}{'cIoU':>8}{'gIoU':>8}{'n':>8}" + (f"{'no mask':>9}" if show_no_mask else "")
    lines = [header]
    for group, values in results.items():
        line = f"{group:<{width}}{values['ciou']:>8.2f}{values['giou']:>8.2f}{values['n']:>8d}"
        if show_no_mask:
            line += f"{values.get('no_mask', 0):>9d}"
        lines.append(line)
    return "\n".join(lines)


def available_sets(cfg: Config, split: str, sets: Sequence[str]) -> Tuple[List[str], List[str]]:
    """``(present, missing)``: which of the instruction ``sets`` have files for ``split`` on disk."""
    present = [s for s in sets if cfg.instruction_files(split, s)]
    return present, [s for s in sets if s not in present]
