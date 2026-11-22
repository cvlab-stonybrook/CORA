"""Anchor mask and bounding-box encoders for conditional instructions (paper Sec. 3.3, SegLLM style).

A conditional query such as "Segment the cars on the left of <REF>" carries its anchor as two LLM input
positions, the SegLLM ``<mask> <box>`` pair (supplement Sec. 5): one embedding of the masked anchor crop and
one embedding of the anchor's bounding box, "encoded separately using an anchor image encoder and a bounding
box encoder as proposed by Seg-LLM".

* Mask embedding: CLS token of CLIP's penultimate layer for the masked, tightly cropped anchor
  (``Batch.ref_pixels``; the CLIP tower is LLaVA's frozen one), mapped to the LLM width by a projector:
  ``separate`` = a trainable copy of LLaVA's multimodal projector (the default), ``shared`` =
  LLaVA's projector itself (SegLLM code).
* Box embedding: ``nn.Linear(4, hidden)`` on ``[y0, x0, y1, x1] / sam_image_size`` (``Batch.ref_boxes``).
* Optional modality dropout of the SegLLM code (drop each embedding with probability p in training, scale by
  ``1 - p`` at test time); it is not described in either paper and is off by default.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
from torch import nn

from cora.tokens import TokenSpec


class AnchorEncoders(nn.Module):
    """Maps CLIP CLS features of anchor crops and anchor boxes to LLM input embeddings."""

    def __init__(
        self,
        hidden_size: int,
        projector: nn.Module,
        mask_projector: str = "separate",
        modality_dropout: float = 0.0,
        train_projector: bool = False,
    ) -> None:
        super().__init__()
        if mask_projector not in ("separate", "shared"):
            raise ValueError(f"mask_projector must be 'separate' or 'shared', got {mask_projector!r}")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError(f"modality_dropout must be in [0, 1), got {modality_dropout}")
        self.mask_projector_mode = mask_projector
        self.modality_dropout = float(modality_dropout)
        self.train_projector = bool(train_projector)
        self.box_encoder = nn.Linear(4, hidden_size)
        if mask_projector == "separate":
            # Initialised from LLaVA's projector; full-precision weights because it is trained.
            self.projector: Optional[nn.Module] = copy.deepcopy(projector).float()
            self.projector.requires_grad_(True)
            self._shared: List[nn.Module] = []
        else:
            self.projector = None
            # Held in a plain list so that LLaVA's projector is not registered (and saved) twice.
            self._shared = [projector]
        # Initial state is "trainable": enables LLaVA's own projector too when it is shared and train_projector is set.
        self.set_trainable(True)

    @property
    def active_projector(self) -> nn.Module:
        return self.projector if self.projector is not None else self._shared[0]

    def own_parameters(self) -> List[nn.Parameter]:
        """Parameters that belong to the anchor encoders and are toggled by :meth:`set_trainable`."""
        params = list(self.box_encoder.parameters())
        if self.projector is not None:
            params += list(self.projector.parameters())
        elif self.train_projector:
            params += list(self._shared[0].parameters())
        return params

    def set_trainable(self, flag: bool) -> None:
        """Enable or disable gradients of the anchor encoders (they train only in stages with references)."""
        for param in self.own_parameters():
            param.requires_grad_(flag)

    def zero_term(self) -> Optional[torch.Tensor]:
        """A zero scalar that depends on every trainable anchor parameter, or None if none is trainable.

        Adding it to the graph of a batch without references keeps these parameters in the reducer of DDP.
        """
        terms = [p.sum() * 0.0 for p in self.own_parameters() if p.requires_grad]
        return torch.stack(terms).sum() if terms else None

    def _drop(self, embedding: torch.Tensor) -> torch.Tensor:
        p = self.modality_dropout
        if p == 0.0:
            return embedding
        if self.training:
            keep = (torch.rand(embedding.shape[0], 1, device=embedding.device) >= p).to(embedding.dtype)
            return embedding * keep
        return embedding * (1.0 - p)

    def forward(self, clip_cls: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``clip_cls`` (R, clip_hidden), ``boxes`` (R, 4) -> mask and box embeddings, each (R, hidden)."""
        if clip_cls.shape[0] != boxes.shape[0]:
            raise ValueError(f"{clip_cls.shape[0]} anchor crops but {boxes.shape[0]} anchor boxes")
        mask_embedding = self.active_projector(clip_cls)
        box_embedding = self.box_encoder(boxes.to(self.box_encoder.weight.dtype))
        return self._drop(mask_embedding), self._drop(box_embedding)


def place_anchor_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    spec: TokenSpec,
    ref_seq: torch.Tensor,
    mask_embeddings: torch.Tensor,
    box_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Overwrite the ``<REFMASK>`` / ``<REFBOX>`` positions with the anchor embeddings (Sec. 3.3).

    The r-th reference goes to the r-th placeholder of its sequence ``ref_seq[r]``; references are listed in
    sequence order, then in order of their placeholders. The embeddings replace the placeholder embeddings
    (SegLLM's released code accidentally adds to them; overwriting is the intended behaviour).
    """
    mask_positions = input_ids == spec.ref_mask_token_id
    box_positions = input_ids == spec.ref_box_token_id
    num_refs = int(ref_seq.shape[0])
    if mask_embeddings.shape[0] != num_refs or box_embeddings.shape[0] != num_refs:
        raise ValueError("one mask and one box embedding are needed per reference")
    if num_refs > 1 and bool((ref_seq[1:] < ref_seq[:-1]).any()):
        raise ValueError("Batch.ref_seq must be sorted by sequence (references in order of their placeholders)")
    expected = torch.bincount(ref_seq, minlength=input_ids.shape[0])
    for name, positions in (("REF_MASK", mask_positions), ("REF_BOX", box_positions)):
        found = positions.sum(1)
        if found.shape != expected.shape or not torch.equal(found, expected):
            raise ValueError(
                f"{name} placeholders per sequence {found.tolist()} do not match the references per sequence "
                f"{expected.tolist()} given by Batch.ref_seq"
            )
    if num_refs == 0:
        return inputs_embeds
    inputs_embeds = inputs_embeds.masked_scatter(mask_positions.unsqueeze(-1), mask_embeddings.to(inputs_embeds.dtype))
    return inputs_embeds.masked_scatter(box_positions.unsqueeze(-1), box_embeddings.to(inputs_embeds.dtype))
