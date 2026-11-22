"""Tests of cora.data.conversation: prompt format, reference rendering, tokenisation, label masking, truncation."""
from functools import lru_cache

import pytest
import torch

from cora import constants as C
from cora.config import Config
from cora.data.conversation import ASSISTANT_MARK, build_prompt, encode, render_query
from cora.tokens import setup_tokenizer

LLAVA = "llava-hf/llava-1.5-7b-hf"
SYSTEM = Config().model.system_prompt
ANSWER = f"It is {C.SEG_TOKEN}."


@lru_cache(maxsize=None)
def get_tokenizer(image_seq_len: int = 576, clip_image_size: int = 336):
    """The real LLaVA tokenizer with the CORA tokens (from the local Hugging Face cache)."""
    try:
        return setup_tokenizer(LLAVA, image_seq_len, clip_image_size)
    except OSError:
        pytest.skip("llava-hf/llava-1.5-7b-hf tokenizer is not in the local Hugging Face cache")


# ----------------------------------------------------------------------------------------------
# Text level
# ----------------------------------------------------------------------------------------------
def test_build_prompt_format():
    prompt = build_prompt("Segment the car.", ANSWER, "SYS")
    assert prompt == "SYS USER: <image>\nSegment the car. ASSISTANT: It is <SSEG>.</s>"
    assert build_prompt("Segment the car.", None, "SYS") == "SYS USER: <image>\nSegment the car. ASSISTANT:"


QUERY = "Segment the cars on the left of <REF>."


def test_render_query_modes():
    assert render_query(QUERY, "tokens", "inline") == "Segment the cars on the left of <REFMASK> <REFBOX>."
    assert render_query(QUERY, "tokens", "prefix") == "<REFMASK> <REFBOX> Segment the cars on the left of the reference object."
    assert render_query(QUERY, "text", "inline") == "Segment the cars on the left of the mask."
    assert render_query(QUERY, "text", "prefix") == "Segment the cars on the left of the mask."
    plain = "Can you segment the road in this image?"
    for mode in ("tokens", "text"):
        for placement in ("inline", "prefix"):
            assert render_query(plain, mode, placement) == plain


def test_render_query_errors():
    with pytest.raises(ValueError, match="reference mode"):
        render_query(QUERY, "pixels", "inline")
    with pytest.raises(ValueError, match="placement"):
        render_query(QUERY, "tokens", "middle")
    with pytest.raises(ValueError, match="at most once"):
        render_query("<REF> and <REF>", "tokens", "inline")


# ----------------------------------------------------------------------------------------------
# Token level
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [4, 576])
def test_encode_single_image_block_and_label_mask(seq_len):
    tok, spec = get_tokenizer(seq_len)
    ids, labels = encode(tok, spec, "Can you segment the car in this image?", ANSWER, SYSTEM, 384)
    assert ids.dtype == labels.dtype == torch.long and ids.shape == labels.shape and ids.dim() == 1
    assert ids[0] == spec.bos_token_id
    positions = (ids == spec.image_token_id).nonzero().flatten()
    assert len(positions) == seq_len
    assert torch.equal(positions, torch.arange(positions[0], positions[0] + seq_len))  # one contiguous block
    assert (ids == spec.seg_token_id).sum() == 1 and ids[-1] == spec.eos_token_id
    supervised = labels != C.IGNORE_INDEX
    assert torch.equal(labels[supervised], ids[supervised])
    # exactly the answer and the closing EOS are supervised
    assert tok.convert_ids_to_tokens(ids[supervised].tolist()) == ["▁It", "▁is", "▁", "<SSEG>", "▁.", "</s>"]
    first = int(supervised.nonzero()[0])
    assert tok.convert_ids_to_tokens(ids[first - 5 : first].tolist()) == ["▁A", "SS", "IST", "ANT", ":"]
    assert supervised.sum() == len(ids) - first  # supervision is one trailing run


def test_encode_matches_tokenizer_on_the_text_without_image_expansion():
    tok, spec = get_tokenizer(4)
    query = "Please segment the traffic light in this image."
    ids, _ = encode(tok, spec, query, ANSWER, SYSTEM, 384)
    plain = tok(build_prompt(query, ANSWER, SYSTEM))["input_ids"]
    expanded = [t for t in ids.tolist() if t != spec.image_token_id]
    assert expanded == [t for t in plain if t != spec.image_token_id]
    assert len(ids) == len(plain) + 3


def test_encode_answer_variants_supervise_only_the_answer():
    tok, spec = get_tokenizer(4)
    for answer in (f"Sure, the segmentation result is {C.SEG_TOKEN}.", f"{C.SEG_TOKEN}."):
        ids, labels = encode(tok, spec, "Segment the sky.", answer, SYSTEM, 384)
        tokens = tok.convert_ids_to_tokens(ids[labels != C.IGNORE_INDEX].tolist())
        assert C.SEG_TOKEN in tokens and tokens[-1] == "</s>"
        assert "USER" not in "".join(tokens) and "ASSISTANT" not in "".join(tokens)


def test_encode_generation_prompt_has_no_answer_and_no_supervision():
    tok, spec = get_tokenizer(4)
    ids, labels = encode(tok, spec, "Segment the sky.", None, SYSTEM, 384)
    assert torch.all(labels == C.IGNORE_INDEX)
    assert tok.convert_ids_to_tokens(ids[-5:].tolist()) == ["▁A", "SS", "IST", "ANT", ":"]
    assert (ids == spec.seg_token_id).sum() == 0


@pytest.mark.parametrize("mode,placement", [("tokens", "inline"), ("tokens", "prefix"), ("text", "inline")])
def test_encode_reference_rendering(mode, placement):
    tok, spec = get_tokenizer(4)
    query = render_query(QUERY, mode, placement)
    ids, labels = encode(tok, spec, query, ANSWER, SYSTEM, 384)
    n_mask, n_box = int((ids == spec.ref_mask_token_id).sum()), int((ids == spec.ref_box_token_id).sum())
    if mode == "tokens":
        assert (n_mask, n_box) == (1, 1)
        mask_pos, box_pos = int((ids == spec.ref_mask_token_id).nonzero()[0]), int((ids == spec.ref_box_token_id).nonzero()[0])
        assert mask_pos < box_pos
        image_end = int((ids == spec.image_token_id).nonzero()[-1])
        if placement == "prefix":
            assert mask_pos < image_end + 6  # right after the image, before the query text
        else:
            assert mask_pos > image_end + 6
    else:
        assert (n_mask, n_box) == (0, 0)
        assert "▁mask" in tok.convert_ids_to_tokens(ids.tolist())
    assert torch.all(labels[(ids == spec.ref_mask_token_id) | (ids == spec.ref_box_token_id)] == C.IGNORE_INDEX)


# ----------------------------------------------------------------------------------------------
# Truncation
# ----------------------------------------------------------------------------------------------
def text_length(ids, spec):
    return int((ids != spec.image_token_id).sum())


def test_truncation_shortens_the_query_only():
    tok, spec = get_tokenizer(4)
    long_query = "Segment " + " ".join(f"object{i}" for i in range(300)) + " in this image."
    ids, labels = encode(tok, spec, long_query, ANSWER, SYSTEM, 120)
    assert text_length(ids, spec) <= 120
    assert tok.convert_ids_to_tokens(ids[labels != C.IGNORE_INDEX].tolist()) == ["▁It", "▁is", "▁", "<SSEG>", "▁.", "</s>"]
    assert (ids == spec.seg_token_id).sum() == 1 and (ids == spec.image_token_id).sum() == 4
    short_ids, _ = encode(tok, spec, "Segment the sky.", ANSWER, SYSTEM, 120)
    assert text_length(short_ids, spec) < 120  # short queries are untouched
    untruncated, _ = encode(tok, spec, "Segment the sky.", ANSWER, SYSTEM, 10_000)
    assert torch.equal(short_ids, untruncated)


def test_truncation_keeps_reference_placeholders():
    tok, spec = get_tokenizer(4)
    query = "<REFMASK> <REFBOX> Segment " + " ".join(f"thing{i}" for i in range(200)) + " near <REFMASK> <REFBOX> end"
    ids, _ = encode(tok, spec, query, ANSWER, SYSTEM, 150)
    assert text_length(ids, spec) <= 150
    assert (ids == spec.ref_mask_token_id).sum() == 2 and (ids == spec.ref_box_token_id).sum() == 2


def test_truncation_exact_budget_is_untouched():
    tok, spec = get_tokenizer(4)
    ids, _ = encode(tok, spec, "Segment the sky.", ANSWER, SYSTEM, 1000)
    budget = text_length(ids, spec)
    same, _ = encode(tok, spec, "Segment the sky.", ANSWER, SYSTEM, budget)
    assert torch.equal(ids, same)
    with pytest.raises(ValueError):
        encode(tok, spec, "Segment the sky.", ANSWER, SYSTEM, budget - 10)  # the fixed part alone no longer fits


# ----------------------------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------------------------
def test_encode_errors():
    tok, spec = get_tokenizer(4)
    with pytest.raises(ValueError, match="exactly one <image>"):
        encode(tok, spec, "Segment <image> the sky.", ANSWER, SYSTEM, 384)
    with pytest.raises(ValueError, match="exactly one <SSEG>"):
        encode(tok, spec, "Segment the sky.", "It is here.", SYSTEM, 384)
    with pytest.raises(ValueError, match="exactly one <SSEG>"):
        encode(tok, spec, "Segment the sky.", "<SSEG> and <SSEG>.", SYSTEM, 384)
    with pytest.raises(ValueError, match="exactly one <SSEG>"):
        encode(tok, spec, "Segment <SSEG> the sky.", ANSWER, SYSTEM, 384)
    with pytest.raises(ValueError, match="max_text_tokens"):
        encode(tok, spec, "Segment the sky.", ANSWER, SYSTEM, 0)


def test_assistant_mark_constant():
    assert ASSISTANT_MARK == "ASSISTANT:"


# ----------------------------------------------------------------------------------------------
# LISA's prompt layout (a tokenizer with <im_start> / <im_end>: converted LISA checkpoints)
# ----------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_lisa_tokenizer(image_seq_len: int = 4, clip_image_size: int = 224):
    """The CORA tokenizer with LISA's two image-boundary tokens, saved and reloaded as tools/convert_lisa.py does."""
    import tempfile

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(LLAVA, padding_side="right")
    except OSError:
        pytest.skip("llava-hf/llava-1.5-7b-hf tokenizer is not in the local Hugging Face cache")
    tokenizer.add_tokens([C.SEG_TOKEN, C.REF_MASK_TOKEN, C.REF_BOX_TOKEN, C.IM_START_TOKEN, C.IM_END_TOKEN], special_tokens=True)
    folder = tempfile.mkdtemp(prefix="cora_lisa_tok_")
    tokenizer.save_pretrained(folder)
    return setup_tokenizer(folder, image_seq_len, clip_image_size)


def test_plain_tokenizer_keeps_the_llava_layout():
    _, spec = get_tokenizer()
    assert not spec.lisa_layout and spec.im_start_token_id is None and spec.im_end_token_id is None


def test_lisa_layout_matches_the_ids_lisa_was_trained_on():
    """LISA's tokenizer stripped the text around added tokens: nothing between ``USER:`` and ``<im_start>``, no
    newline after ``<im_end>``, and the word before the segmentation token directly in front of it."""
    tokenizer, spec = get_lisa_tokenizer()
    assert spec.lisa_layout and len({spec.im_start_token_id, spec.im_end_token_id, spec.seg_token_id, spec.image_token_id}) == 4
    ids, labels = encode(tokenizer, spec, "Can you segment the car in this image?", ANSWER, SYSTEM, 384)
    ids = ids.tolist()
    colon, word = tokenizer.convert_tokens_to_ids(":"), tokenizer.convert_tokens_to_ids
    start = ids.index(spec.im_start_token_id)
    assert ids[0] == spec.bos_token_id and ids[start - 1] == colon and tokenizer.convert_ids_to_tokens(ids[start - 2]) == "ER"
    assert ids[start + 1 : start + 1 + spec.image_seq_len] == [spec.image_token_id] * spec.image_seq_len
    end = start + 1 + spec.image_seq_len
    assert ids[end] == spec.im_end_token_id and ids[end + 1] == word("▁Can"), tokenizer.convert_ids_to_tokens(ids[end : end + 3])
    seg = ids.index(spec.seg_token_id)
    assert ids[seg - 2 : seg] == [word("▁It"), word("▁is")] and ids[seg + 1 :] == [word("▁."), spec.eos_token_id]
    assert ids.count(spec.image_token_id) == spec.image_seq_len and ids.count(spec.seg_token_id) == 1
    # the text reads as LISA's conversation (special tokens aside)
    text = tokenizer.decode([i for i in ids if i != spec.image_token_id], skip_special_tokens=True)
    assert " ".join(text.split()) == f"{SYSTEM} USER: Can you segment the car in this image? ASSISTANT: It is ."
    # supervision: the answer and the closing EOS only
    supervised = [i for i, l in zip(ids, labels.tolist()) if l != C.IGNORE_INDEX]
    assert supervised == [word("▁It"), word("▁is"), spec.seg_token_id, word("▁."), spec.eos_token_id]
    assert all(l == i for i, l in zip(ids, labels.tolist()) if l != C.IGNORE_INDEX)


def test_lisa_layout_answer_variants_generation_and_references():
    tokenizer, spec = get_lisa_tokenizer()
    colon = tokenizer.convert_tokens_to_ids(":")
    # ReasonSeg's validation answer "<SSEG>.": the token follows "ASSISTANT:" directly; LISA's leading space in the
    # validation query makes no difference in this layout
    a, _ = encode(tokenizer, spec, " the thing that holds water Please output segmentation mask.", f"{C.SEG_TOKEN}.", SYSTEM, 384)
    b, _ = encode(tokenizer, spec, "the thing that holds water Please output segmentation mask.", f"{C.SEG_TOKEN}.", SYSTEM, 384)
    assert a.tolist() == b.tolist() and a.tolist()[a.tolist().index(spec.seg_token_id) - 1] == colon
    # generation prompt: ends at "ASSISTANT:", nothing supervised, and it is a prefix of the full conversation
    prompt, labels = encode(tokenizer, spec, "Segment the bus.", None, SYSTEM, 384)
    full, _ = encode(tokenizer, spec, "Segment the bus.", f"Sure, it is {C.SEG_TOKEN}.", SYSTEM, 384)
    assert prompt.tolist()[-1] == colon and bool((labels == C.IGNORE_INDEX).all()) and full.tolist()[: len(prompt)] == prompt.tolist()
    # anchor placeholders of a conditional query stay in the sequence, once each
    query = render_query(QUERY, "tokens", "inline")
    ids, _ = encode(tokenizer, spec, query, ANSWER, SYSTEM, 384)
    assert (ids == spec.ref_mask_token_id).sum() == 1 and (ids == spec.ref_box_token_id).sum() == 1
    # a query that is too long is shortened, the answer is not
    long_ids, _ = encode(tokenizer, spec, "word " * 600, ANSWER, SYSTEM, 64)
    whole, _ = encode(tokenizer, spec, "word " * 600, ANSWER, SYSTEM, 10_000)
    assert len(long_ids) - spec.image_seq_len == 64 and len(whole) > len(long_ids) + 400  # the budget is met exactly
    assert long_ids.tolist()[-6:] == whole.tolist()[-6:] and long_ids.tolist()[-1] == spec.eos_token_id
    assert (long_ids == spec.seg_token_id).sum() == 1
    with pytest.raises(ValueError, match="exactly once"):
        encode(tokenizer, spec, "Segment the bus.", "It is here.", SYSTEM, 384)
