"""Collation of samples into a :class:`cora.data.batch.Batch` (paper Sec. 3.1-3.3; LISA ``collate_fn``).

Every conversation becomes one sequence: prompt rendered with :mod:`cora.data.conversation`, right padded.
Sequences of one image are contiguous and keep the order of ``Sample.conversations``. For unlabeled samples
the K-1 rephrasings of each trained query form a second batch (``Batch.aux``, kind ``"aux"``) over the same
images; ``aux.group[j]`` is the index of the trained sequence aux sequence ``j`` belongs to (Sec. 3.2.1).
"""
from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from cora import constants as C
from cora.config import Config
from cora.data.batch import Batch, Conversation, Sample
from cora.data.conversation import encode, render_query
from cora.tokens import TokenSpec

KINDS = ("labeled", "unlabeled", "eval")


def _pad(rows: Sequence[torch.Tensor], value: int) -> torch.Tensor:
    """Right-pad 1-D tensors to the longest one."""
    out = torch.full((len(rows), max(int(r.numel()) for r in rows)), value, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : row.numel()] = row
    return out


def _encode_all(
    queries: Sequence[str], answers: Sequence[Optional[str]], tokenizer, spec: TokenSpec, cfg: Config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(input_ids, attention_mask, labels), each (N, L), for the given stored queries and answers.

    An answer of ``None`` gives a prompt that ends in ``ASSISTANT:`` (for generation)."""
    mode, placement = cfg.model.reference.mode, cfg.model.reference.placement
    ids, labels = [], []
    for query, answer in zip(queries, answers):
        rendered = render_query(query, mode, placement)
        i, l = encode(tokenizer, spec, rendered, answer, cfg.model.system_prompt, cfg.data.max_text_tokens)
        ids.append(i)
        labels.append(l)
    input_ids = _pad(ids, spec.pad_token_id)
    attention = _pad([torch.ones_like(i) for i in ids], 0)
    return input_ids, attention, _pad(labels, C.IGNORE_INDEX)


def _references(
    flat: Sequence[Conversation], spec: TokenSpec, cfg: Config, input_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference tensors in sequence order; checks that every placeholder pair has its anchor inputs."""
    size = spec.clip_image_size
    pixels: List[torch.Tensor] = []
    boxes: List[torch.Tensor] = []
    seq: List[int] = []
    for n, conv in enumerate(flat):
        n_mask = int((input_ids[n] == spec.ref_mask_token_id).sum())
        n_box = int((input_ids[n] == spec.ref_box_token_id).sum())
        if (n_mask, n_box) != ((1, 1) if C.REF_TOKEN in conv.query and cfg.model.reference.mode == "tokens" else (0, 0)):
            raise ValueError(f"sequence {n}: {n_mask} mask / {n_box} box placeholders for query {conv.query!r}")
        if n_mask:
            if conv.ref_pixels is None or conv.ref_box is None:
                raise ValueError(f"sequence {n}: the query has {C.REF_TOKEN} but the conversation has no ref_pixels / ref_box")
            pixels.append(conv.ref_pixels)
            boxes.append(conv.ref_box)
            seq.append(n)
    if not pixels:
        return torch.zeros(0, 3, size, size), torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    return torch.stack(pixels), torch.stack(boxes).float(), torch.tensor(seq, dtype=torch.long)


def collate(samples: List[Sample], tokenizer, spec: TokenSpec, cfg: Config, generation: bool = False) -> Batch:
    """Build a :class:`Batch` from samples of one kind (labeled, unlabeled or eval).

    For unlabeled samples every conversation must carry its ``aux_queries``; they are encoded with the answer
    of the trained query and form ``batch.aux``. With ``generation=True`` (evaluation samples only) the
    prompts stop at ``ASSISTANT:`` so that the model writes the answer itself (``CoraModel.generate``).
    """
    if not samples:
        raise ValueError("cannot collate an empty list of samples")
    kind = samples[0].kind
    if kind not in KINDS or any(s.kind != kind for s in samples):
        raise ValueError(f"samples must all be of one kind in {KINDS}, got {sorted({s.kind for s in samples})}")
    flat: List[Conversation] = []
    seq_image: List[int] = []
    for b, sample in enumerate(samples):
        if not sample.conversations:
            raise ValueError(f"sample {sample.image_id} has no conversation")
        flat.extend(sample.conversations)
        seq_image.extend([b] * len(sample.conversations))
    if generation and kind != "eval":
        raise ValueError("generation prompts are built for evaluation samples only")
    answers = [None if generation else c.answer for c in flat]
    input_ids, attention, labels = _encode_all([c.query for c in flat], answers, tokenizer, spec, cfg)
    ref_pixels, ref_boxes, ref_seq = _references(flat, spec, cfg, input_ids)
    images_sam = torch.stack([s.image_sam for s in samples])
    images_clip = torch.stack([s.image_clip for s in samples])
    batch = Batch(
        kind=kind,
        image_ids=[s.image_id for s in samples],
        images_sam=images_sam,
        images_clip=images_clip,
        resize_hw=[tuple(s.resize_hw) for s in samples],
        orig_hw=[tuple(s.orig_hw) for s in samples],
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        seq_image=torch.tensor(seq_image, dtype=torch.long),
        masks=[c.mask for c in flat],
        class_ids=torch.tensor([c.class_id for c in flat], dtype=torch.long),
        set_names=[c.set_name for c in flat],
        ref_pixels=ref_pixels,
        ref_boxes=ref_boxes,
        ref_seq=ref_seq,
        confidence=[c.confidence for c in flat] if kind == "unlabeled" else None,
        queries=[c.query for c in flat],
        answers=[c.answer for c in flat],
        record_ids=[c.record_id for c in flat],
        is_long_query=[c.is_long_query for c in flat],
    )
    if kind == "unlabeled":
        batch.aux = _aux_batch(batch, flat, tokenizer, spec, cfg)
    return batch


def _aux_batch(parent: Batch, flat: Sequence[Conversation], tokenizer, spec: TokenSpec, cfg: Config) -> Batch:
    """The K-1 rephrasings of every trained query as one batch sharing the parent's images (Sec. 3.2.1)."""
    queries: List[str] = []
    answers: List[str] = []
    group: List[int] = []
    for n, conv in enumerate(flat):
        if not conv.aux_queries:
            raise ValueError(f"unlabeled conversation {conv.record_id} has no aux_queries")
        for query in conv.aux_queries:
            queries.append(query)
            answers.append(conv.answer)
            group.append(n)
    input_ids, attention, labels = _encode_all(queries, answers, tokenizer, spec, cfg)
    index = torch.tensor(group, dtype=torch.long)
    size = spec.clip_image_size
    return Batch(
        kind="aux",
        image_ids=parent.image_ids,
        images_sam=parent.images_sam,
        images_clip=parent.images_clip,
        resize_hw=parent.resize_hw,
        orig_hw=parent.orig_hw,
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        seq_image=parent.seq_image[index],
        masks=[parent.masks[g] for g in group],
        class_ids=parent.class_ids[index],
        set_names=[parent.set_names[g] for g in group],
        ref_pixels=torch.zeros(0, 3, size, size),
        ref_boxes=torch.zeros(0, 4),
        ref_seq=torch.zeros(0, dtype=torch.long),
        group=index,
        queries=queries,
        answers=answers,
        record_ids=[None] * len(group),
        is_long_query=[None] * len(group),
    )


def make_collate(tokenizer, spec: TokenSpec, cfg: Config, generation: bool = False) -> Callable[[List[Sample]], Batch]:
    """Picklable collate function for ``DataLoader(collate_fn=...)``."""
    return partial(collate, tokenizer=tokenizer, spec=spec, cfg=cfg, generation=generation)
