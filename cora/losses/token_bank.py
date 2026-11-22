"""Token bank of labeled-image <SSEG> features (Sec. 3.2.2): one FIFO queue per class."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class TokenBank(nn.Module):
    """Per-class FIFO of detached <SSEG> features of labeled images.

    The queues live in registered buffers, so ``state_dict()`` saves and restores them. Entries are never reset
    between epochs; the newest entries overwrite the oldest ones of their class.
    """

    def __init__(self, num_classes: int, dim: int, size: int) -> None:
        super().__init__()
        if num_classes < 1 or dim < 1 or size < 1:
            raise ValueError(f"num_classes, dim and size must be positive, got {num_classes}, {dim}, {size}")
        self.num_classes = num_classes
        self.dim = dim
        self.size = size
        self.register_buffer("features", torch.zeros(num_classes, size, dim))
        self.register_buffer("filled", torch.zeros(num_classes, dtype=torch.long))  # valid slots per class
        self.register_buffer("next_slot", torch.zeros(num_classes, dtype=torch.long))  # write position per class

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor, class_ids: torch.Tensor) -> None:
        """Append ``(n, dim)`` features under their class ids; ids < 0 (unknown class) are ignored.

        Within one call later rows are newer. If a class receives more rows than ``size`` only the newest ``size``
        are kept. The features are detached and stored in float32.
        """
        if features.ndim != 2 or features.shape[1] != self.dim:
            raise ValueError(f"features must have shape (n, {self.dim}), got {tuple(features.shape)}")
        if class_ids.shape != (features.shape[0],):
            raise ValueError(f"class_ids must have shape ({features.shape[0]},), got {tuple(class_ids.shape)}")
        if features.shape[0] == 0:
            return
        class_ids = class_ids.to(self.features.device).long()
        top = int(class_ids.max())
        if top >= self.num_classes:
            raise ValueError(f"class id {top} is out of range for a bank of {self.num_classes} classes")
        features = features.detach().to(device=self.features.device, dtype=self.features.dtype)
        for c in class_ids[class_ids >= 0].unique().tolist():
            rows = features[class_ids == c][-self.size:]
            n = rows.shape[0]
            slots = (self.next_slot[c] + torch.arange(n, device=rows.device)) % self.size
            self.features[c, slots] = rows
            self.next_slot[c] = (self.next_slot[c] + n) % self.size
            self.filled[c] = min(int(self.filled[c]) + n, self.size)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the stored ``(M, dim)`` features and their ``(M,)`` class ids (filled slots only).

        Classes appear in ascending order, each from its oldest to its newest entry.
        """
        parts, labels = [], []
        for c in range(self.num_classes):
            n = int(self.filled[c])
            if n == 0:
                continue
            slots = (self.next_slot[c] - n + torch.arange(n, device=self.features.device)) % self.size
            parts.append(self.features[c, slots])
            labels.append(torch.full((n,), c, dtype=torch.long, device=self.features.device))
        if not parts:
            return (self.features.new_zeros(0, self.dim),
                    torch.zeros(0, dtype=torch.long, device=self.features.device))
        return torch.cat(parts), torch.cat(labels)

    def count(self, class_id: int) -> int:
        """Number of entries currently stored for ``class_id``."""
        if not 0 <= class_id < self.num_classes:
            raise ValueError(f"class id {class_id} is out of range for a bank of {self.num_classes} classes")
        return int(self.filled[class_id])
