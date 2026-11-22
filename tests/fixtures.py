"""Shared test fixtures: a tiny CORA model, its tokenizer, and valid :class:`Batch` objects.

Importable on its own (it needs only the frozen modules and ``cora.model``). The tiny model is a random
LLaVA (4 image tokens per 28 px image) plus a random two-block SAM on 64 px images; the tokenizer is the real
LLaVA tokenizer from the Hugging Face cache (a few MB).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch

from cora import constants as C
from cora.config import Config
from cora.data.batch import Batch
from cora.model.build import build_model, build_tokenizer
from cora.model.cora import CoraModel
from cora.tokens import TokenSpec

DEFAULT_ANSWER = f"It is {C.SEG_TOKEN}."
TINY_SAM_SIZE = 64


def tiny_config(**model_overrides) -> Config:
    """Configuration of the tiny random model: fp32 on CPU, 64 px SAM frame, gradient checkpointing on."""
    cfg = Config()
    cfg.model.tiny = True
    cfg.model.precision = "fp32"
    cfg.data.sam_image_size = TINY_SAM_SIZE
    for key, value in model_overrides.items():
        setattr(cfg.model, key, value)
    return cfg


def tiny_model_and_tokenizer(cfg: Optional[Config] = None, seed: int = 0) -> Tuple[CoraModel, object]:
    """``(model, tokenizer)`` with the tokenizer already attached to the model; ``model.spec`` is the TokenSpec."""
    cfg = cfg or tiny_config()
    torch.manual_seed(seed)
    tokenizer, spec = build_tokenizer(cfg)
    model = build_model(cfg, spec)
    model.attach_tokenizer(tokenizer)
    return model, tokenizer


@dataclass
class ImageInput:
    """One image of a batch, already preprocessed (used to build batches from real frames)."""

    sam: torch.Tensor  # (3, S, S)
    clip: torch.Tensor  # (3, C, C)
    resize_hw: Tuple[int, int]
    orig_hw: Tuple[int, int]
    image_id: str = "image"


def sam_resize_shape(orig_hw: Tuple[int, int], size: int) -> Tuple[int, int]:
    """Size of an image after SAM's ResizeLongestSide (longest side -> ``size``)."""
    scale = size / max(orig_hw)
    return int(orig_hw[0] * scale + 0.5), int(orig_hw[1] * scale + 0.5)


def encode_prompts(
    tokenizer, spec: TokenSpec, system_prompt: str, queries: Sequence[str], answer: Optional[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenise ``"<system> USER: <image>\\n<query> ASSISTANT: <answer></s>"`` for every query.

    The image token is repeated ``spec.image_seq_len`` times; labels are IGNORE_INDEX up to and including
    ``ASSISTANT:``. With ``answer=None`` the prompts end at ``ASSISTANT:`` (generation) and every label is
    ignored. Returns right-padded ``(input_ids, attention_mask, labels)``.
    """
    rows, label_rows = [], []
    for query in queries:
        prefix = f"{system_prompt} USER: {C.IMAGE_TOKEN}\n{query} ASSISTANT:"
        text = prefix if answer is None else f"{prefix} {answer}</s>"
        ids = tokenizer(text)["input_ids"]
        prefix_ids = tokenizer(prefix)["input_ids"]
        if ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(f"tokenisation of the prompt prefix is not a prefix of the full prompt: {text!r}")
        if ids.count(spec.image_token_id) != 1:
            raise ValueError(f"expected exactly one {C.IMAGE_TOKEN} in {text!r}")
        at = ids.index(spec.image_token_id)
        expand = [spec.image_token_id] * (spec.image_seq_len - 1)
        ids = ids[:at] + expand + ids[at:]
        labels = list(ids)
        for i in range(len(prefix_ids) + spec.image_seq_len - 1):
            labels[i] = C.IGNORE_INDEX
        rows.append(ids)
        label_rows.append(labels)
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), spec.pad_token_id, dtype=torch.long)
    labels = torch.full((len(rows), width), C.IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros(len(rows), width, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(rows, label_rows)):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attention[i, : len(ids)] = 1
    return input_ids, attention, labels


def default_query(index: int, with_reference: bool) -> str:
    """A short query whose length varies with ``index`` (so that batches need padding)."""
    words = " ".join(["thing"] * (index % 3 + 1))
    if with_reference:
        return f"Segment the {words} on the left of {C.REF_MASK_TOKEN} {C.REF_BOX_TOKEN} please"
    return f"Segment the {words} number {index}"


def _random_mask(hw: Tuple[int, int], gen: torch.Generator) -> torch.Tensor:
    h, w = hw
    mask = torch.zeros(h, w, dtype=torch.uint8)
    y0, x0 = int(torch.randint(0, h // 2, (1,), generator=gen)), int(torch.randint(0, w // 2, (1,), generator=gen))
    mask[y0 : y0 + h // 3 + 1, x0 : x0 + w // 3 + 1] = 1
    return mask


def make_batch(
    cfg: Config,
    tokenizer,
    spec: TokenSpec,
    *,
    kind: str = "labeled",
    seqs_per_image: Union[int, Sequence[int]] = (2,),
    queries: Optional[Sequence[str]] = None,
    ref_sequences: Sequence[int] = (),
    answer: Optional[str] = DEFAULT_ANSWER,
    aux_per_seq: int = 0,
    images: Optional[Sequence[ImageInput]] = None,
    masks: Optional[Sequence[torch.Tensor]] = None,
    references: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
    orig_hw: Tuple[int, int] = (40, 60),
    seed: int = 0,
) -> Batch:
    """A valid :class:`Batch`.

    ``seqs_per_image`` gives the number of sequences of each image (an int means one image). ``queries`` has one
    entry per sequence; the default queries are generated, and the sequences listed in ``ref_sequences`` get a
    ``<REFMASK> <REFBOX>`` pair. One reference (random anchor crop and box, or the given ``references``) is
    created per ``<REFMASK>`` found in the prompts, in sequence order. ``kind="unlabeled"`` adds a confidence map
    per sequence and, when ``aux_per_seq`` (K-1) is positive, an aux batch of K-1 rephrasings per sequence.
    ``answer=None`` builds prompts that end in ``ASSISTANT:``.
    """
    if kind not in ("labeled", "unlabeled", "eval"):
        raise ValueError(f"kind must be labeled, unlabeled or eval, got {kind!r}")
    counts = [seqs_per_image] if isinstance(seqs_per_image, int) else list(seqs_per_image)
    num_images, num_seqs = len(counts), sum(counts)
    gen = torch.Generator().manual_seed(seed)
    sam_size, clip_size = cfg.data.sam_image_size, spec.clip_image_size

    if images is None:
        inputs = []
        for b in range(num_images):
            hw = (orig_hw[0] + 4 * b, orig_hw[1])
            resize = sam_resize_shape(hw, sam_size)
            sam = torch.randn(3, sam_size, sam_size, generator=gen)
            sam[:, resize[0] :, :] = 0
            sam[:, :, resize[1] :] = 0
            inputs.append(ImageInput(sam, torch.randn(3, clip_size, clip_size, generator=gen), resize, hw, f"image{b}"))
    else:
        inputs = list(images)
        if len(inputs) != num_images:
            raise ValueError(f"{len(inputs)} images given but seqs_per_image lists {num_images}")

    if queries is None:
        queries = [default_query(i, i in set(ref_sequences)) for i in range(num_seqs)]
    elif len(queries) != num_seqs:
        raise ValueError(f"{len(queries)} queries given for {num_seqs} sequences")
    input_ids, attention, labels = encode_prompts(tokenizer, spec, cfg.model.system_prompt, queries, answer)

    seq_image = torch.repeat_interleave(torch.arange(num_images), torch.tensor(counts))
    if masks is None:
        masks = [_random_mask(inputs[int(seq_image[i])].orig_hw, gen) for i in range(num_seqs)]
    refs_per_seq = (input_ids == spec.ref_mask_token_id).sum(1)
    ref_seq = torch.repeat_interleave(torch.arange(num_seqs), refs_per_seq)
    num_refs = int(ref_seq.numel())
    if references is None:
        ref_pixels = torch.randn(num_refs, 3, clip_size, clip_size, generator=gen)
        ref_boxes = torch.rand(num_refs, 4, generator=gen)
    else:
        if len(references) != num_refs:
            raise ValueError(f"{len(references)} references given but the prompts contain {num_refs} placeholders")
        ref_pixels = torch.stack([r[0] for r in references]) if references else torch.zeros(0, 3, clip_size, clip_size)
        ref_boxes = torch.stack([r[1] for r in references]) if references else torch.zeros(0, 4)

    set_names = [
        C.SET_UNLABELED if kind == "unlabeled" else (C.SET_CONDITIONAL if refs_per_seq[i] else C.SET_SEMANTIC)
        for i in range(num_seqs)
    ]
    batch = Batch(
        kind=kind,
        image_ids=[im.image_id for im in inputs],
        images_sam=torch.stack([im.sam for im in inputs]),
        images_clip=torch.stack([im.clip for im in inputs]),
        resize_hw=[im.resize_hw for im in inputs],
        orig_hw=[im.orig_hw for im in inputs],
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        seq_image=seq_image,
        masks=list(masks),
        class_ids=torch.arange(num_seqs) % 3,
        set_names=set_names,
        ref_pixels=ref_pixels,
        ref_boxes=ref_boxes,
        ref_seq=ref_seq,
        queries=list(queries),
        answers=[answer or ""] * num_seqs,
        record_ids=[None] * num_seqs,
        is_long_query=[None] * num_seqs,
    )
    if kind == "unlabeled":
        batch.confidence = [
            torch.rand(*inputs[int(seq_image[i])].orig_hw, generator=gen) for i in range(num_seqs)
        ]
        if aux_per_seq > 0:
            batch.aux = _make_aux(cfg, tokenizer, spec, batch, aux_per_seq)
    return batch


def _make_aux(cfg: Config, tokenizer, spec: TokenSpec, parent: Batch, per_seq: int) -> Batch:
    """The K-1 rephrasings of every sequence of ``parent`` as a batch over the same images."""
    num_seqs = parent.num_sequences
    group = torch.repeat_interleave(torch.arange(num_seqs), per_seq)
    queries = [f"Please find the objects of type {g} variant {k % per_seq}" for k, g in enumerate(group.tolist())]
    input_ids, attention, labels = encode_prompts(tokenizer, spec, cfg.model.system_prompt, queries, DEFAULT_ANSWER)
    n_aux = int(group.numel())
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
        seq_image=parent.seq_image[group],
        masks=[parent.masks[int(g)] for g in group],
        class_ids=parent.class_ids[group],
        set_names=[C.SET_UNLABELED] * n_aux,
        ref_pixels=torch.zeros(0, 3, spec.clip_image_size, spec.clip_image_size),
        ref_boxes=torch.zeros(0, 4),
        ref_seq=torch.zeros(0, dtype=torch.long),
        group=group,
        queries=queries,
        answers=[DEFAULT_ANSWER] * n_aux,
        record_ids=[None] * n_aux,
        is_long_query=[None] * n_aux,
    )
