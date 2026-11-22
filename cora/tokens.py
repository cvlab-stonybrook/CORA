"""Tokenizer set-up shared by the data pipeline and the model.

The LLaVA-1.5 tokenizer gets three extra tokens: ``<SSEG>`` (the segmentation token of the paper) and
the two placeholders that carry the anchor mask embedding and the anchor box embedding of a conditional
query. The LLaVA-1.5 embedding matrix has spare rows (32064 rows for a 32002-token vocabulary), so the
new ids fit without resizing; ``cora.model`` initialises their rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from cora import constants as C


@dataclass(frozen=True)
class TokenSpec:
    """Token ids and image-token geometry needed to build and to read sequences."""

    image_token_id: int
    seg_token_id: int
    ref_mask_token_id: int
    ref_box_token_id: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    image_seq_len: int  # LLM positions taken by one image (576 for CLIP ViT-L/14-336)
    clip_image_size: int  # side of the CLIP input in pixels (336)
    vocab_size: int  # len(tokenizer) after adding the tokens
    # Only for a tokenizer that has LISA's <im_start> / <im_end> (a converted LISA checkpoint, tools/convert_lisa.py):
    # the image features are then wrapped by the two tokens and the prompt is tokenised as LISA does.
    im_start_token_id: Optional[int] = None
    im_end_token_id: Optional[int] = None

    @property
    def lisa_layout(self) -> bool:
        return self.im_start_token_id is not None


def setup_tokenizer(name_or_path: str, image_seq_len: int = 576, clip_image_size: int = 336) -> Tuple[object, TokenSpec]:
    """Load the LLaVA tokenizer, add the CORA tokens and return ``(tokenizer, TokenSpec)``.

    Right padding is used for training batches. BOS is prepended by the tokenizer; EOS is not appended
    automatically, so prompts spell out ``</s>``.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, padding_side="right")
    tokenizer.add_tokens([C.SEG_TOKEN, C.REF_MASK_TOKEN, C.REF_BOX_TOKEN], special_tokens=True)

    def token_id(token: str) -> int:
        ids = tokenizer.convert_tokens_to_ids(token)
        if ids is None or ids == tokenizer.unk_token_id:
            raise ValueError(f"token {token!r} is missing from the tokenizer")
        return int(ids)

    def optional_id(token: str) -> Optional[int]:
        ids = tokenizer.convert_tokens_to_ids(token)
        return None if ids is None or ids == tokenizer.unk_token_id else int(ids)

    im_start, im_end = optional_id(C.IM_START_TOKEN), optional_id(C.IM_END_TOKEN)
    if (im_start is None) != (im_end is None):
        raise ValueError(f"the tokenizer {name_or_path} has only one of {C.IM_START_TOKEN} and {C.IM_END_TOKEN}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token
    spec = TokenSpec(
        image_token_id=token_id(C.IMAGE_TOKEN),
        seg_token_id=token_id(C.SEG_TOKEN),
        ref_mask_token_id=token_id(C.REF_MASK_TOKEN),
        ref_box_token_id=token_id(C.REF_BOX_TOKEN),
        pad_token_id=int(tokenizer.pad_token_id),
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        image_seq_len=int(image_seq_len),
        clip_image_size=int(clip_image_size),
        vocab_size=len(tokenizer),
        im_start_token_id=im_start,
        im_end_token_id=im_end,
    )
    return tokenizer, spec
