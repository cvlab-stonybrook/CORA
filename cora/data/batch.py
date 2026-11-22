"""In-memory containers passed between the datasets, the collate function, the model and the losses.

This module is the contract between ``cora.data`` and ``cora.model`` / ``cora.losses`` / ``cora.engine``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple

import torch


@dataclass
class Conversation:
    """One query/answer pair about one image, with its target mask."""

    query: str  # text without the image token; may contain REF_TOKEN once
    answer: str  # contains SEG_TOKEN exactly once (e.g. "It is <SSEG>.")
    mask: torch.Tensor  # (H, W) uint8 at original resolution: 1 target, 0 background, 255 ignored
    class_id: int  # class of the target, or -1 when unknown
    set_name: str  # semantic | attribute | conditional | unlabeled | reasonseg
    # Reference (anchor) object of a conditional query, already preprocessed (cora.data.transforms):
    ref_pixels: Optional[torch.Tensor] = None  # (3, C, C) CLIP-normalised masked anchor crop
    ref_box: Optional[torch.Tensor] = None  # (4,) anchor box [y0, x0, y1, x1] / sam_image_size in the SAM frame
    # Unlabeled images only: the other K-1 rephrasings of the same class query (Sec. 3.2.1)
    aux_queries: Optional[List[str]] = None
    # Unlabeled images only: per-pixel confidence of the semi-supervised model, (H, W) float in [0, 1]
    confidence: Optional[torch.Tensor] = None
    record_id: Optional[str] = None
    is_long_query: Optional[bool] = None  # ReasonSeg only


@dataclass
class Sample:
    """One image with the conversations sampled for it."""

    image_id: str
    image_sam: torch.Tensor  # (3, S, S) float32, SAM-normalised, zero padded bottom/right
    image_clip: torch.Tensor  # (3, C, C) float32, CLIP-normalised
    resize_hw: Tuple[int, int]  # size of the image inside the SAM frame before padding
    orig_hw: Tuple[int, int]  # original image size (H, W)
    conversations: List[Conversation]
    kind: str = "labeled"  # labeled | unlabeled | eval


@dataclass
class Batch:
    """A collated batch. ``B`` images, ``N`` sequences (conversations), ``R`` reference objects.

    Sequences belonging to one image are contiguous and keep the order of ``Sample.conversations``.
    Every sequence contains exactly one ``<SSEG>`` token (in its answer).
    """

    kind: str  # labeled | unlabeled | eval | aux
    image_ids: List[str]  # B
    images_sam: torch.Tensor  # (B, 3, S, S)
    images_clip: torch.Tensor  # (B, 3, C, C)
    resize_hw: List[Tuple[int, int]]  # B
    orig_hw: List[Tuple[int, int]]  # B
    input_ids: torch.Tensor  # (N, L) long; the image placeholder is already repeated image_seq_len times
    attention_mask: torch.Tensor  # (N, L) long, 1 for real tokens (right padding)
    labels: torch.Tensor  # (N, L) long; IGNORE_INDEX everywhere except the answer tokens and the closing EOS
    seq_image: torch.Tensor  # (N,) long, image index of every sequence
    masks: List[torch.Tensor]  # N tensors (H, W) uint8 with values 0 / 1 / 255(ignored)
    class_ids: torch.Tensor  # (N,) long, -1 when unknown
    set_names: List[str]  # N
    ref_pixels: torch.Tensor  # (R, 3, C, C); R may be 0
    ref_boxes: torch.Tensor  # (R, 4)
    ref_seq: torch.Tensor  # (R,) long: sequence each reference belongs to, in order of its placeholders
    # Unlabeled batches: the K-1 other rephrasings as a second batch over the same images
    # (kind="aux", N*(K-1) sequences); aux.group[j] is the index of the sequence in this batch that
    # aux sequence j belongs to. The aux batch shares images_sam / images_clip with its parent.
    aux: Optional["Batch"] = None
    group: Optional[torch.Tensor] = None  # (N_aux,) long, only on aux batches
    confidence: Optional[List[Optional[torch.Tensor]]] = None  # N entries (H, W) float, unlabeled only
    queries: List[str] = field(default_factory=list)  # N, for logging
    answers: List[str] = field(default_factory=list)  # N, for logging
    record_ids: List[Optional[str]] = field(default_factory=list)  # N
    is_long_query: List[Optional[bool]] = field(default_factory=list)  # N

    @property
    def num_images(self) -> int:
        return len(self.image_ids)

    @property
    def num_sequences(self) -> int:
        return int(self.input_ids.shape[0])

    def to(self, device, non_blocking: bool = True) -> "Batch":
        """Move every tensor (also inside lists and the aux batch) to ``device``; returns self."""

        def move(value):
            if torch.is_tensor(value):
                return value.to(device, non_blocking=non_blocking)
            if isinstance(value, Batch):
                return value.to(device, non_blocking=non_blocking)
            if isinstance(value, list):
                return [move(v) for v in value]
            return value

        for f in fields(self):
            setattr(self, f.name, move(getattr(self, f.name)))
        return self

    def pin_memory(self) -> "Batch":
        def pin(value):
            if torch.is_tensor(value):
                return value.pin_memory()
            if isinstance(value, Batch):
                return value.pin_memory()
            if isinstance(value, list):
                return [pin(v) for v in value]
            return value

        for f in fields(self):
            setattr(self, f.name, pin(getattr(self, f.name)))
        return self
