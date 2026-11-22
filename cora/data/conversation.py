"""Prompt construction and tokenisation (paper Sec. 3.1, Sec. 3.3; supplement Sec. 5).

Every query is an independent single-turn conversation in LISA's ``llava_v1`` format::

    <system prompt> USER: <image>\\n<query> ASSISTANT: <answer></s>

The image placeholder is expanded to ``spec.image_seq_len`` ids, the labels supervise only the answer and the
closing ``</s>`` (LISA's collate rule; the system prompt, the user turn and ``ASSISTANT:`` are ignored).
Reference (anchor) objects of conditional queries are written either as the pair ``<REFMASK> <REFBOX>``
whose two positions carry the anchor mask and box embeddings (supplement Sec. 5, "<mask> <box>"), or as plain
text for models without the anchor encoders.

The LLaVA tokenizer prepends BOS, writes a ``"▁"`` token around special tokens and does not append EOS, so special tokens are always
located by id, never by string offsets.

With a tokenizer that has ``<im_start>`` / ``<im_end>`` (a converted LISA checkpoint; ``TokenSpec.lisa_layout``)
the ids are built as LISA's code builds them for its released checkpoints, which were trained on exactly this
layout (``mm_use_im_start_end``; the tokenizer of that time stripped the text around added tokens)::

    <system prompt> USER:<im_start><image><im_end><query> ASSISTANT: <text before the token><SSEG><text after></s>

that is, no ``"▁"`` token before ``<im_start>`` or before ``<SSEG>`` and no newline after ``<im_end>``; the hidden
state that predicts the segmentation token is then the one LISA trained (the last word before the token).
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import torch

from cora import constants as C
from cora.tokens import TokenSpec

LOG = logging.getLogger(__name__)

REFERENCE_MODES = ("tokens", "text")
PLACEMENTS = ("inline", "prefix")
REFERENCE_PHRASE = "the reference object"  # what REF_TOKEN becomes in the text of a "prefix" query
EOS_TEXT = "</s>"
ASSISTANT_MARK = "ASSISTANT:"


def render_query(query: str, reference_mode: str, placement: str) -> str:
    """Replace the anchor placeholder of a stored query for the chosen reference mode.

    * ``"tokens"`` + ``"inline"``: ``<REF>`` becomes ``<REFMASK> <REFBOX>`` where it stands;
    * ``"tokens"`` + ``"prefix"``: the pair is put before the query text (SegLLM style) and ``<REF>`` in the
      text becomes "the reference object";
    * ``"text"``: ``<REF>`` becomes "the mask" (``constants.REF_TEXT_FALLBACK``); placement is irrelevant.

    Queries without ``<REF>`` are returned unchanged. At most one anchor per query is supported.
    """
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f"unknown reference mode {reference_mode!r}; expected one of {REFERENCE_MODES}")
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown reference placement {placement!r}; expected one of {PLACEMENTS}")
    count = query.count(C.REF_TOKEN)
    if count == 0:
        return query
    if count > 1:
        raise ValueError(f"a query may contain {C.REF_TOKEN} at most once, got {count}: {query!r}")
    if reference_mode == "text":
        return query.replace(C.REF_TOKEN, C.REF_TEXT_FALLBACK)
    pair = f"{C.REF_MASK_TOKEN} {C.REF_BOX_TOKEN}"
    if placement == "inline":
        return query.replace(C.REF_TOKEN, pair)
    return f"{pair} {query.replace(C.REF_TOKEN, REFERENCE_PHRASE)}"


def build_prompt(query: str, answer: str | None, system_prompt: str) -> str:
    """``"<system> USER: <image>\\n<query> ASSISTANT: <answer></s>"``; with ``answer=None`` the text ends with
    ``"ASSISTANT:"`` (generation prompt)."""
    head = f"{system_prompt} USER: {C.IMAGE_TOKEN}\n{query} {ASSISTANT_MARK}"
    if answer is None:
        return head
    return f"{head} {answer}{EOS_TEXT}"


def _tokenize(tokenizer, text: str) -> List[int]:
    """Ids of ``text`` with BOS (and without EOS), as the LLaVA tokenizer produces them."""
    return list(tokenizer(text, add_special_tokens=True)["input_ids"])


def _plain(tokenizer, text: str) -> List[int]:
    """Ids of a piece of text without BOS (empty for an empty piece)."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else []


def _lisa_ids(tokenizer, spec: TokenSpec, query: str, answer: str | None, system_prompt: str) -> List[int]:
    """Ids of one conversation in the layout of LISA's released checkpoints (see the module docstring), with a
    single image placeholder. The pieces between the special tokens are tokenised one by one, stripped, which is
    what the tokenizer of LISA's code did around added tokens."""
    ids = _tokenize(tokenizer, f"{system_prompt} USER:")
    ids += [spec.im_start_token_id, spec.image_token_id, spec.im_end_token_id]
    ids += _plain(tokenizer, f"{query.strip()} {ASSISTANT_MARK}")
    if answer is None:
        return ids
    before, token, after = answer.partition(C.SEG_TOKEN)
    if not token or C.SEG_TOKEN in after:
        raise ValueError(f"the answer must contain {C.SEG_TOKEN} exactly once: {answer!r}")
    return ids + _plain(tokenizer, before.strip()) + [spec.seg_token_id] + _plain(tokenizer, after.strip()) + [spec.eos_token_id]


def _conversation_ids(tokenizer, spec: TokenSpec, query: str, answer: str | None, system_prompt: str) -> List[int]:
    """Ids of one conversation with a single image placeholder, in the layout the tokenizer calls for."""
    if spec.lisa_layout:
        return _lisa_ids(tokenizer, spec, query, answer, system_prompt)
    return _tokenize(tokenizer, build_prompt(query, answer, system_prompt))


def _drop_tail_tokens(ids: Sequence[int], count: int, protected: frozenset) -> List[int]:
    """Remove ``count`` unprotected ids from the end of ``ids``; protected ids always stay."""
    kept: List[int] = []
    remaining = count
    for token_id in reversed(ids):
        if remaining > 0 and token_id not in protected:
            remaining -= 1
            continue
        kept.append(token_id)
    kept.reverse()
    return kept


def _shorten_query(tokenizer, spec: TokenSpec, query: str, excess: int) -> str:
    """The query with ``excess`` text tokens removed from its end (the reference placeholders are kept)."""
    query_ids = list(tokenizer(query, add_special_tokens=False)["input_ids"])
    protected = frozenset((spec.ref_mask_token_id, spec.ref_box_token_id))
    if sum(1 for t in query_ids if t not in protected) < excess:
        raise ValueError("the query cannot be shortened enough: the fixed part of the prompt alone exceeds max_text_tokens")
    kept = _drop_tail_tokens(query_ids, excess, protected)
    return tokenizer.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()


def encode(
    tokenizer, spec: TokenSpec, query: str, answer: str | None, system_prompt: str, max_text_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenise one conversation: ``(input_ids, labels)``, 1-D long tensors.

    The single image placeholder is repeated ``spec.image_seq_len`` times. ``labels`` equal ``input_ids``
    on the answer tokens and the closing EOS and are ``IGNORE_INDEX`` elsewhere (everything is ignored when
    ``answer`` is None). A query that makes the text longer than ``max_text_tokens`` (image tokens not
    counted) is shortened at its end; the answer, EOS and reference placeholders are never cut.

    Raises ``ValueError`` unless the sequence has exactly one image block and, when an answer is given,
    exactly one ``<SSEG>`` token and the closing EOS.
    """
    if max_text_tokens < 1:
        raise ValueError(f"max_text_tokens must be positive, got {max_text_tokens}")
    ids = _conversation_ids(tokenizer, spec, query, answer, system_prompt)
    excess = len(ids) - 1 - max_text_tokens  # the single image placeholder is not a text token
    if excess > 0:
        original_refs = (ids.count(spec.ref_mask_token_id), ids.count(spec.ref_box_token_id))
        for _ in range(excess + 8):  # re-tokenising a shortened query can shift the count by a token or two
            query = _shorten_query(tokenizer, spec, query, excess)
            ids = _conversation_ids(tokenizer, spec, query, answer, system_prompt)
            excess = len(ids) - 1 - max_text_tokens
            if excess <= 0:
                break
        else:
            raise ValueError(f"could not shorten the query to {max_text_tokens} text tokens")
        if (ids.count(spec.ref_mask_token_id), ids.count(spec.ref_box_token_id)) != original_refs:
            raise ValueError("shortening the query changed its reference placeholders")
        LOG.debug("query shortened to fit %d text tokens", max_text_tokens)

    if ids.count(spec.image_token_id) != 1:
        raise ValueError(f"expected exactly one {C.IMAGE_TOKEN} in the prompt, found {ids.count(spec.image_token_id)}")
    prefix = _conversation_ids(tokenizer, spec, query, None, system_prompt)
    if ids[: len(prefix)] != prefix:
        raise RuntimeError("the tokenised prompt prefix is not a prefix of the tokenised conversation")
    image_pos = ids.index(spec.image_token_id)
    ids = ids[:image_pos] + [spec.image_token_id] * spec.image_seq_len + ids[image_pos + 1 :]
    answer_start = len(prefix) + spec.image_seq_len - 1
    if answer is not None:
        if ids.count(spec.seg_token_id) != 1:
            raise ValueError(f"expected exactly one {C.SEG_TOKEN} in the conversation, found {ids.count(spec.seg_token_id)}")
        if ids[-1] != spec.eos_token_id:
            raise RuntimeError("the conversation does not end with the EOS token")
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.full_like(input_ids, C.IGNORE_INDEX)
    if answer is not None:
        labels[answer_start:] = input_ids[answer_start:]
    return input_ids, labels
