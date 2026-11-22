"""Optimizer and learning-rate schedule (paper Sec. 4.1: AdamW, learning rate 2e-5; the rest follows LISA)."""
from __future__ import annotations

from typing import Iterable

import torch

from cora.config import TrainConfig


def build_optimizer(parameters: Iterable[torch.nn.Parameter], tcfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW over the given trainable parameters (LISA: betas (0.9, 0.95), no weight decay)."""
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(params, lr=tcfg.lr, betas=tuple(tcfg.betas), weight_decay=tcfg.weight_decay)


def lr_factor(step: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    """Multiplier of the base learning rate before optimizer step ``step`` (0-based).

    ``warmup_linear_decay`` is LISA's WarmupDecayLR: linear warm-up from 0 over ``warmup_steps``, then
    linear decay to 0 at ``total_steps``. ``constant`` keeps the base rate after the warm-up.
    """
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def build_scheduler(optimizer: torch.optim.Optimizer, tcfg: TrainConfig, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, tcfg.warmup_steps, tcfg.schedule)
    )
