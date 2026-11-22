"""CPU tests of the CORA network on the tiny random LLaVA + SAM (cora.model.cora / build / llava / sam)."""
from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from cora import constants as C
from cora.model.llava import clip_cls_features, ensure_embedding_rows, image_features
from tests.fixtures import make_batch, tiny_config, tiny_model_and_tokenizer

TRAINABLE_PREFIXES = (
    "llava.model.language_model.embed_tokens",
    "llava.lm_head",
    "text_hidden_fcs",
    "sam.mask_decoder",
    "anchors",
)
FROZEN_PREFIXES = (
    "llava.model.vision_tower",
    "llava.model.multi_modal_projector",
    "sam.image_encoder",
    "sam.prompt_encoder",
)


@pytest.fixture(scope="module")
def tiny():
    model, tokenizer = tiny_model_and_tokenizer()
    model.eval()
    return model, tokenizer


def batch_for(model, tokenizer, **kwargs):
    return make_batch(model.cfg, tokenizer, model.spec, **kwargs)


def capture_hidden(model):
    """Forward hook that records the language model's last hidden state."""
    store = []
    handle = model.llava.model.language_model.register_forward_hook(lambda m, a, out: store.append(out.last_hidden_state))
    return store, handle


# ----------------------------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------------------------
def test_trainable_and_frozen_parameters():
    model, _ = tiny_model_and_tokenizer()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    for name in trainable:
        assert "lora_" in name or name.startswith(TRAINABLE_PREFIXES), name
    for prefix in TRAINABLE_PREFIXES + ("llava.model.language_model.layers.0.self_attn.q_proj.lora_A",):
        assert any(n.startswith(prefix) for n in trainable), prefix
    for name, p in model.named_parameters():
        if name.startswith(FROZEN_PREFIXES) or "iou_prediction_head" in name:
            assert not p.requires_grad, name
    # frozen base weights of the language model stay frozen; embeddings and head are fp32 copies
    base = model.llava.model.language_model.layers[0].self_attn.q_proj.base_layer.weight
    assert not base.requires_grad
    assert model.llava.get_input_embeddings().weight.dtype == torch.float32
    assert model.llava.get_output_embeddings().weight.dtype == torch.float32


def test_lora_only_on_language_model_attention():
    model, _ = tiny_model_and_tokenizer()
    names = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    layers = model.llava.config.text_config.num_hidden_layers
    assert len(names) == 2 * layers
    assert all("language_model.layers" in n and n.endswith(("q_proj", "v_proj")) for n in names)
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    assert lora and all(p.dtype == torch.float32 for p in lora)


def test_new_token_rows_are_mean_of_pretrained_rows():
    model, _ = tiny_model_and_tokenizer()
    spec = model.spec
    for weight in (model.llava.get_input_embeddings().weight, model.llava.get_output_embeddings().weight):
        assert weight.shape[0] == 32064  # never shrunk
        mean = weight[: spec.image_token_id].mean(dim=0)
        for token_id in (spec.seg_token_id, spec.ref_mask_token_id, spec.ref_box_token_id):
            torch.testing.assert_close(weight[token_id], mean, atol=1e-6, rtol=1e-5)


def test_ensure_embedding_rows_grows_but_never_shrinks():
    model, _ = tiny_model_and_tokenizer()
    llava = model.llava
    ensure_embedding_rows(llava, 100)
    assert llava.get_input_embeddings().weight.shape[0] == 32064
    ensure_embedding_rows(llava, 32100)
    assert llava.get_input_embeddings().weight.shape[0] == 32128  # padded to a multiple of 64
    assert llava.get_output_embeddings().weight.shape[0] == 32128


def test_tokenizer_geometry_matches_tiny_llava():
    model, tokenizer = tiny_model_and_tokenizer()
    assert model.spec.image_seq_len == 4 and model.spec.clip_image_size == 28
    assert model.spec.seg_token_id == tokenizer.convert_tokens_to_ids(C.SEG_TOKEN)
    assert model.spec.vocab_size == len(tokenizer)


# ----------------------------------------------------------------------------------------------
# Forward
# ----------------------------------------------------------------------------------------------
def test_forward_shapes_and_dtypes(tiny):
    model, tok = tiny
    batch = batch_for(model, tok, seqs_per_image=(2, 1))
    out = model(batch)
    assert out.ce_loss.shape == () and out.ce_loss.dtype == torch.float32 and torch.isfinite(out.ce_loss)
    assert len(out.mask_logits) == 3
    for n, logits in enumerate(out.mask_logits):
        assert logits.dtype == torch.float32
        assert tuple(logits.shape) == batch.orig_hw[int(batch.seq_image[n])]
    hidden = model.llava.config.text_config.hidden_size
    assert out.seg_hidden.shape == (3, hidden) and out.seg_hidden.dtype == torch.float32
    assert out.seg_embed.shape == (3, model.cfg.model.out_dim) and out.seg_embed.dtype == torch.float32
    assert out.sam_embeddings.shape == (2, 256, 4, 4) and out.sam_embeddings.dtype == torch.float32


def test_compute_text_loss_false_returns_no_ce(tiny):
    model, tok = tiny
    out = model(batch_for(model, tok), compute_text_loss=False)
    assert out.ce_loss is None and len(out.mask_logits) == 2


def test_seg_hidden_is_state_before_the_seg_token(tiny):
    model, tok = tiny
    batch = batch_for(model, tok, seqs_per_image=(2, 1))
    store, handle = capture_hidden(model)
    out = model(batch)
    handle.remove()
    hidden = store[0]
    position = (batch.input_ids == model.spec.seg_token_id).int().argmax(1)
    rows = torch.arange(3)
    torch.testing.assert_close(out.seg_hidden, hidden[rows, position - 1])
    assert not torch.allclose(out.seg_hidden, hidden[rows, position])
    expected_embed = model.text_hidden_fcs(out.seg_hidden)
    torch.testing.assert_close(out.seg_embed, expected_embed)


@pytest.mark.parametrize("bad", ["none", "two"])
def test_exactly_one_seg_token_is_required(tiny, bad):
    model, tok = tiny
    batch = batch_for(model, tok)
    seg = model.spec.seg_token_id
    if bad == "none":
        batch.input_ids[1][batch.input_ids[1] == seg] = 5
    else:
        position = int((batch.input_ids[0] == seg).nonzero()[0])
        batch.input_ids[0, position - 2] = seg
    with pytest.raises(ValueError, match="exactly one"):
        model(batch)


def test_wrong_number_of_image_tokens_raises(tiny):
    model, tok = tiny
    batch = batch_for(model, tok)
    position = int((batch.input_ids[0] == model.spec.image_token_id).nonzero()[0])
    batch.input_ids[0, position] = 5
    with pytest.raises(ValueError, match="image tokens"):
        model(batch)


def test_ce_loss_equals_manual_cross_entropy(tiny):
    model, tok = tiny
    batch = batch_for(model, tok, seqs_per_image=(2, 1))
    store, handle = capture_hidden(model)
    out = model(batch)
    handle.remove()
    logits = model.llava.lm_head(store[0])  # (N, L, V)
    manual = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), batch.labels[:, 1:].reshape(-1), ignore_index=C.IGNORE_INDEX
    )
    torch.testing.assert_close(out.ce_loss, manual)
    assert int((batch.labels != C.IGNORE_INDEX).sum()) > 0
    # prompt, image tokens and padding are never supervised
    assert (batch.labels[batch.input_ids == model.spec.image_token_id] == C.IGNORE_INDEX).all()
    assert (batch.labels[batch.attention_mask == 0] == C.IGNORE_INDEX).all()


def test_all_ignored_labels_give_a_zero_loss(tiny):
    model, tok = tiny
    batch = batch_for(model, tok)
    batch.labels.fill_(C.IGNORE_INDEX)
    out = model(batch)
    assert out.ce_loss.item() == 0.0


def test_several_sequences_share_one_image(tiny):
    model, tok = tiny
    batch = batch_for(model, tok, seqs_per_image=(3,))
    clip_calls, sam_calls = [], []
    h1 = model.llava.model.vision_tower.register_forward_hook(lambda m, a, out: clip_calls.append(a[0].shape[0]))
    h2 = model.sam.image_encoder.register_forward_hook(lambda m, a, out: sam_calls.append(a[0].shape[0]))
    out = model(batch)
    h1.remove()
    h2.remove()
    assert clip_calls == [1] and sam_calls == [1]  # one CLIP / SAM pass for the image, not one per sequence
    assert len(out.mask_logits) == 3
    assert out.sam_embeddings.shape[0] == 1
    # the three prompts differ, so the three masks differ
    assert not torch.allclose(out.mask_logits[0], out.mask_logits[1])


def test_sam_embeddings_can_be_passed_back(tiny):
    model, tok = tiny
    batch = batch_for(model, tok, seqs_per_image=(2, 1))
    first = model(batch)
    calls = []
    handle = model.sam.image_encoder.register_forward_hook(lambda m, a, out: calls.append(1))
    second = model(batch, sam_embeddings=first.sam_embeddings)
    handle.remove()
    assert calls == []
    for a, b in zip(first.mask_logits, second.mask_logits):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="one embedding per image"):
        model(batch, sam_embeddings=first.sam_embeddings[:1])


def test_padding_does_not_change_a_sequence(tiny):
    """A sequence gives the same state alone and inside a padded batch (right padding is masked)."""
    model, tok = tiny
    batch = batch_for(model, tok, seqs_per_image=(1, 1), queries=["Segment the thing", "Segment the thing thing thing thing"])
    assert int(batch.attention_mask[0].sum()) < int(batch.attention_mask[1].sum())
    both = model(batch)
    single = batch_for(model, tok, seqs_per_image=(1,), queries=["Segment the thing"])
    single.images_sam, single.images_clip = batch.images_sam[:1], batch.images_clip[:1]
    alone = model(single)
    torch.testing.assert_close(both.seg_hidden[0], alone.seg_hidden[0], atol=1e-5, rtol=1e-4)


def test_invalid_seq_image_raises(tiny):
    model, tok = tiny
    batch = batch_for(model, tok)
    batch.seq_image = torch.tensor([0, 5])
    with pytest.raises(ValueError, match="seq_image"):
        model(batch)


# ----------------------------------------------------------------------------------------------
# Anchors
# ----------------------------------------------------------------------------------------------
def test_anchor_embeddings_land_on_the_placeholder_positions(tiny):
    model, tok = tiny
    spec = model.spec
    two_refs = f"Segment what lies between {C.REF_MASK_TOKEN} {C.REF_BOX_TOKEN} and {C.REF_MASK_TOKEN} {C.REF_BOX_TOKEN}"
    queries = ["Segment the thing", f"Segment it near {C.REF_MASK_TOKEN} {C.REF_BOX_TOKEN}", "Segment more", two_refs]
    batch = batch_for(model, tok, seqs_per_image=(2, 2), queries=queries)
    assert batch.ref_seq.tolist() == [1, 3, 3]
    captured = []
    handle = model.llava.model.language_model.register_forward_pre_hook(
        lambda m, a, kw: captured.append(kw["inputs_embeds"]), with_kwargs=True
    )
    model(batch)
    handle.remove()
    embeds = captured[0]
    mask_emb, box_emb = model.anchors(clip_cls_features(model.llava, batch.ref_pixels), batch.ref_boxes)
    mask_pos = batch.input_ids == spec.ref_mask_token_id
    box_pos = batch.input_ids == spec.ref_box_token_id
    assert mask_pos.sum() == 3 and box_pos.sum() == 3
    torch.testing.assert_close(embeds[mask_pos], mask_emb)
    torch.testing.assert_close(embeds[box_pos], box_emb)
    # image positions carry the image features of their image, text positions the token embeddings
    features = image_features(model.llava, batch.images_clip)
    image_pos = batch.input_ids == spec.image_token_id
    torch.testing.assert_close(embeds[image_pos], features[batch.seq_image].reshape(-1, features.shape[-1]))
    text_pos = ~(mask_pos | box_pos | image_pos)
    token_emb = model.llava.get_input_embeddings()(batch.input_ids)
    torch.testing.assert_close(embeds[text_pos], token_emb[text_pos])


def test_placeholders_without_matching_references_raise(tiny):
    model, tok = tiny
    batch = batch_for(model, tok, ref_sequences=(0,))
    batch.ref_seq = torch.zeros(0, dtype=torch.long)
    batch.ref_pixels = batch.ref_pixels[:0]
    batch.ref_boxes = batch.ref_boxes[:0]
    with pytest.raises(ValueError, match="do not match"):
        model(batch)


def test_batch_without_references_still_trains_the_anchor_parameters():
    model, tok = tiny_model_and_tokenizer()
    model.train()
    batch = batch_for(model, tok, seqs_per_image=(1, 1))
    assert batch.ref_seq.numel() == 0
    out = model(batch)
    (out.ce_loss + sum(m.mean() for m in out.mask_logits)).backward()
    anchor_params = [p for n, p in model.named_parameters() if n.startswith("anchors") and p.requires_grad]
    assert anchor_params
    for p in anchor_params:  # in the graph (DDP sees them) but with zero gradient
        assert p.grad is not None and float(p.grad.abs().sum()) == 0.0


def test_reference_trainable_toggle_and_state_dict():
    model, tok = tiny_model_and_tokenizer()
    assert any(n.startswith("anchors") for n in model.trainable_state_dict())
    model.set_reference_trainable(False)
    assert not any(n.startswith("anchors") for n in model.trainable_state_dict())
    out = model(batch_for(model, tok, ref_sequences=(0,)))  # references still encoded, no gradients to them
    assert torch.isfinite(out.seg_embed).all()
    model.set_reference_trainable(True)
    assert any(n.startswith("anchors") for n in model.trainable_state_dict())


def test_text_reference_mode_has_no_anchor_encoders(caplog):
    model, tok = tiny_model_and_tokenizer(tiny_config(reference=dataclasses.replace(tiny_config().model.reference, mode="text")))
    assert model.anchors is None
    assert not any(n.startswith("anchors") for n, _ in model.named_parameters())
    out = model(batch_for(model, tok))
    assert torch.isfinite(out.ce_loss)
    with pytest.raises(ValueError, match="no anchor encoders"):
        model(batch_for(model, tok, ref_sequences=(0,)))
    model.set_reference_trainable(True)  # warns, does not fail


def test_shared_projector_is_not_duplicated():
    cfg = tiny_config(reference=dataclasses.replace(tiny_config().model.reference, mask_projector="shared"))
    model, tok = tiny_model_and_tokenizer(cfg)
    assert model.anchors.projector is None
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.startswith("anchors.projector") for n in names)
    assert sum("multi_modal_projector" in n for n in names) == 4  # linear_1/2 weight+bias, once
    assert not model.llava.model.multi_modal_projector.linear_1.weight.requires_grad
    out = model(batch_for(model, tok, ref_sequences=(0,)))
    assert torch.isfinite(out.seg_embed).all()


# ----------------------------------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------------------------------
def test_gradients_reach_exactly_the_trainable_parts():
    model, tok = tiny_model_and_tokenizer()
    model.train()
    with torch.no_grad():  # lora_B starts at zero, which would hide the gradient of lora_A
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.05)
    batch = batch_for(model, tok, seqs_per_image=(2, 2), ref_sequences=(0, 3))
    out = model(batch)
    loss = out.ce_loss + sum(F.binary_cross_entropy_with_logits(m, torch.ones_like(m)) for m in out.mask_logits)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no gradient for trainable {name}"
            assert torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, f"unexpected gradient for frozen {name}"
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    for prefix in (
        "llava.model.language_model.layers.0.self_attn.q_proj.lora_A",
        "llava.model.language_model.layers.1.self_attn.v_proj.lora_B",
        "llava.model.language_model.embed_tokens",
        "llava.lm_head",
        "text_hidden_fcs.0",
        "text_hidden_fcs.2",
        "sam.mask_decoder.transformer",
        "sam.mask_decoder.output_hypernetworks_mlps",
        "anchors.box_encoder",
        "anchors.projector.linear_1",
    ):
        assert any(n.startswith(prefix) and float(g.abs().sum()) > 0 for n, g in grads.items()), prefix
    # the <SSEG> row of the input embedding receives gradient (it is learned, Sec. 3.1)
    embed_grad = grads["llava.model.language_model.embed_tokens.weight"]
    assert float(embed_grad[model.spec.seg_token_id].abs().sum()) > 0


def test_text_loss_off_still_trains_the_decoder_path():
    model, tok = tiny_model_and_tokenizer()
    model.train()
    out = model(batch_for(model, tok), compute_text_loss=False)
    sum(m.mean() for m in out.mask_logits).backward()
    assert any(float(p.grad.abs().sum()) > 0 for n, p in model.named_parameters() if n.startswith("sam.mask_decoder") and p.grad is not None)
    assert model.llava.lm_head.weight.grad is None  # the output head is only used by the LM loss


# ----------------------------------------------------------------------------------------------
# Trainable state dict
# ----------------------------------------------------------------------------------------------
def test_trainable_state_dict_round_trip(tmp_path):
    a, tok = tiny_model_and_tokenizer(seed=1)
    b, _ = tiny_model_and_tokenizer(seed=1)
    a.eval()
    b.eval()
    with torch.no_grad():
        for p in a.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn_like(p))
    batch = batch_for(a, tok, seqs_per_image=(2, 1), ref_sequences=(0,))
    out_a = a(batch)
    assert not torch.allclose(out_a.seg_embed, b(batch).seg_embed)
    state = a.trainable_state_dict()
    assert state and all(torch.is_tensor(v) and v.device.type == "cpu" for v in state.values())
    path = tmp_path / "trainable.pt"
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    b.load_trainable_state_dict(loaded)
    for name, value in b.trainable_state_dict().items():
        assert torch.equal(value, state[name]), name
    out_b = b(batch)
    torch.testing.assert_close(out_a.seg_embed, out_b.seg_embed)
    for x, y in zip(out_a.mask_logits, out_b.mask_logits):
        torch.testing.assert_close(x, y)
    # the saved copy is independent of later updates
    with torch.no_grad():
        next(p for p in a.parameters() if p.requires_grad).add_(1.0)
    assert not torch.equal(a.trainable_state_dict()[next(iter(state))], state[next(iter(state))])


def test_load_trainable_state_dict_strictness(caplog):
    model, _ = tiny_model_and_tokenizer()
    state = model.trainable_state_dict()
    dropped = dict(state)
    dropped.pop("text_hidden_fcs.0.weight")
    with pytest.raises(KeyError, match="missing"):
        model.load_trainable_state_dict(dropped)
    extra = dict(state)
    extra["not.a.parameter"] = torch.zeros(1)
    with pytest.raises(KeyError, match="unexpected"):
        model.load_trainable_state_dict(extra)
    with caplog.at_level("WARNING"):
        model.load_trainable_state_dict(dropped, strict=False)
        model.load_trainable_state_dict(extra, strict=False)
    assert "absent" in caplog.text and "ignoring" in caplog.text
    wrong = dict(state)
    wrong["text_hidden_fcs.0.weight"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="shape"):
        model.load_trainable_state_dict(wrong)
    # a checkpoint that holds the anchor encoders loads into a model whose anchors are frozen, with strict=False
    model.set_reference_trainable(False)
    with pytest.raises(KeyError):
        model.load_trainable_state_dict(state)
    model.load_trainable_state_dict(state, strict=False)


# ----------------------------------------------------------------------------------------------
# generate()
# ----------------------------------------------------------------------------------------------
def prompts(model, tok, **kwargs):
    kwargs.setdefault("seqs_per_image", (2, 1))
    return batch_for(model, tok, answer=None, **kwargs)


def scripted_head(model, token_ids):
    """Make the output head produce the given token at generation step i (a stand-in for a trained model)."""
    calls = []

    def hook(module, args, out):
        step = len(calls)
        calls.append(step)
        fake = torch.zeros_like(out)
        fake[:, token_ids[min(step, len(token_ids) - 1)]] = 100.0
        return fake

    return model.llava.lm_head.register_forward_hook(hook)


def test_generate_returns_text_and_masks(tiny):
    model, tok = tiny
    batch = prompts(model, tok)
    texts, masks = model.generate(batch, max_new_tokens=6)
    assert len(texts) == 3 and all(isinstance(t, str) for t in texts)
    assert len(masks) == 3
    for n, mask in enumerate(masks):
        assert mask is None or tuple(mask.shape) == batch.orig_hw[int(batch.seq_image[n])]


def test_generate_decodes_the_mask_of_the_first_seg_token(tiny):
    model, tok = tiny
    spec = model.spec
    batch = prompts(model, tok)
    handle = scripted_head(model, [5000, spec.seg_token_id, spec.eos_token_id])
    texts, masks = model.generate(batch, max_new_tokens=10)
    handle.remove()
    assert all(C.SEG_TOKEN in t for t in texts)
    assert all(mask is not None for mask in masks)
    # the same masks as a teacher-forced forward over prompt + the scripted answer
    forced_ids, forced_mask = [], []
    for n in range(3):
        length = int(batch.attention_mask[n].sum())
        forced_ids.append(torch.cat([batch.input_ids[n, :length], torch.tensor([5000, spec.seg_token_id, spec.eos_token_id])]))
    width = max(len(i) for i in forced_ids)
    ids = torch.full((3, width), spec.pad_token_id)
    attention = torch.zeros(3, width, dtype=torch.long)
    for n, row in enumerate(forced_ids):
        ids[n, : len(row)] = row
        attention[n, : len(row)] = 1
    forced = dataclasses.replace(batch, input_ids=ids, attention_mask=attention, labels=torch.full_like(ids, C.IGNORE_INDEX))
    out = model(forced, compute_text_loss=False)
    for got, want in zip(masks, out.mask_logits):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_generate_marks_sequences_without_seg_as_none(tiny):
    model, tok = tiny
    handle = scripted_head(model, [5000, model.spec.eos_token_id])
    texts, masks = model.generate(prompts(model, tok), max_new_tokens=10)
    handle.remove()
    assert masks == [None, None, None]
    assert all(C.SEG_TOKEN not in t for t in texts)


def test_generate_only_cuts_at_the_first_seg_token(tiny):
    model, tok = tiny
    seg = model.spec.seg_token_id
    handle = scripted_head(model, [seg, seg, 5000, model.spec.eos_token_id])
    texts, masks = model.generate(prompts(model, tok, seqs_per_image=(1,)), max_new_tokens=10)
    handle.remove()
    assert texts[0].count(C.SEG_TOKEN) == 2 and masks[0] is not None


def test_cached_left_padded_greedy_matches_plain_recomputation(tiny):
    model, tok = tiny
    batch = prompts(model, tok, seqs_per_image=(2, 1))
    assert len(set(batch.attention_mask.sum(1).tolist())) > 1  # prompts of different lengths -> padding
    steps = 5
    embeds = model._input_embeddings(batch, batch.input_ids)
    with torch.no_grad():
        got = model._greedy(embeds, batch.attention_mask, steps)
        for n in range(3):
            length = int(batch.attention_mask[n].sum())
            current = embeds[n : n + 1, :length]
            expected = []
            for _ in range(steps):
                hidden = model._run_llm(current, torch.ones(1, current.shape[1], dtype=torch.long))
                logits = model.llava.lm_head(hidden[:, -1]).float()
                for banned in (model.spec.image_token_id, model.spec.ref_mask_token_id, model.spec.ref_box_token_id,
                               model.spec.pad_token_id, model.spec.bos_token_id):
                    logits[:, banned] = float("-inf")
                logits[:, model.spec.vocab_size :] = float("-inf")
                token = int(logits.argmax())
                if token == model.spec.eos_token_id:
                    break
                expected.append(token)
                current = torch.cat([current, model.llava.get_input_embeddings()(torch.tensor([[token]]))], dim=1)
            assert got[n] == expected, (n, got[n], expected)


def test_generate_restores_training_mode_and_builds_no_graph():
    model, tok = tiny_model_and_tokenizer()
    model.train()
    flags = {n: m.training for n, m in model.named_modules()}
    batch = prompts(model, tok)
    handle = scripted_head(model, [5000, model.spec.seg_token_id, model.spec.eos_token_id])
    _, masks = model.generate(batch, max_new_tokens=5)
    handle.remove()
    assert flags == {n: m.training for n, m in model.named_modules()}
    assert all(m is not None and not m.requires_grad for m in masks)


def test_generate_rejects_prompts_that_already_hold_an_answer(tiny):
    model, tok = tiny
    with pytest.raises(ValueError, match="without an answer"):
        model.generate(batch_for(model, tok), max_new_tokens=3)


# ----------------------------------------------------------------------------------------------
# Precision
# ----------------------------------------------------------------------------------------------
def test_bf16_forward_and_backward_on_cpu():
    cfg = tiny_config(precision="bf16")
    model, tok = tiny_model_and_tokenizer(cfg)
    model.train()
    assert model.dtype == torch.bfloat16
    assert model.llava.get_input_embeddings().weight.dtype == torch.float32  # full-precision copy
    out = model(batch_for(model, tok, ref_sequences=(0,)))
    assert out.seg_hidden.dtype == torch.float32 and out.mask_logits[0].dtype == torch.float32
    assert torch.isfinite(out.ce_loss)
    (out.ce_loss + out.mask_logits[0].mean()).backward()
    assert model.llava.get_input_embeddings().weight.grad.dtype == torch.float32
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ----------------------------------------------------------------------------------------------
# Builders and SAM helpers
# ----------------------------------------------------------------------------------------------
def test_embeddings_stay_frozen_when_train_embeddings_is_off():
    model, tok = tiny_model_and_tokenizer(tiny_config(train_embeddings=False))
    assert not model.llava.get_input_embeddings().weight.requires_grad
    assert not model.llava.get_output_embeddings().weight.requires_grad
    out = model(batch_for(model, tok))
    assert torch.isfinite(out.ce_loss)


def test_checkpoint_holds_full_embeddings_and_head_by_default():
    """Fig. 3 draws the MLLM frozen apart from LoRA; like LISA, CORA also trains embed_tokens and lm_head in full."""
    model, _ = tiny_model_and_tokenizer()
    state = model.trainable_state_dict()
    embed = model.llava.get_input_embeddings().weight
    head = model.llava.get_output_embeddings().weight
    assert head.requires_grad and embed.requires_grad
    # the whole matrices (all rows), not only the new <SSEG> rows; nothing else of LLaVA besides LoRA
    assert sum(t.shape == embed.shape for t in state.values()) == 2
    assert len([n for n in state if n.startswith("llava.") and "lora_" not in n]) == 2
    off, _ = tiny_model_and_tokenizer(tiny_config(train_embeddings=False))
    assert all("lora_" in n for n in off.trainable_state_dict() if n.startswith("llava."))


def test_seg_projection_can_be_frozen():
    """Fig. 3 marks only LoRA and the decoder as trained; with both switches off nothing else of the model trains."""
    model, tok = tiny_model_and_tokenizer()
    assert all(p.requires_grad for p in model.text_hidden_fcs.parameters())
    assert len([n for n in model.trainable_state_dict() if n.startswith("text_hidden_fcs.")]) == 4

    model, tok = tiny_model_and_tokenizer(tiny_config(train_seg_projection=False, train_embeddings=False))
    assert not any(p.requires_grad for p in model.text_hidden_fcs.parameters())
    model.set_reference_trainable(False)  # a stage without references, as stage 1
    groups = {"lora" if "lora_" in n else n.split(".")[0] + "." + n.split(".")[1] for n in model.trainable_state_dict()}
    assert groups == {"lora", "sam.mask_decoder"}
    before = [p.detach().clone() for p in model.text_hidden_fcs.parameters()]
    out = model(batch_for(model, tok))
    assert torch.isfinite(out.ce_loss)
    (out.ce_loss + out.mask_logits[0].float().mean()).backward()
    assert all(p.grad is None for p in model.text_hidden_fcs.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for n, p in model.named_parameters() if n.startswith("sam.mask_decoder."))
    assert all(torch.equal(a, b) for a, b in zip(before, model.text_hidden_fcs.parameters()))


def test_mask_decoder_can_be_frozen():
    model, _ = tiny_model_and_tokenizer(tiny_config(train_mask_decoder=False))
    assert not any(p.requires_grad for p in model.sam.parameters())


def test_sam_builders_reject_bad_settings(tmp_path):
    from cora.model.sam import build_sam, tiny_sam

    with pytest.raises(ValueError, match="multiple of 16"):
        tiny_sam(70)
    cfg = tiny_config()
    cfg.model.tiny = False
    cfg.paths.sam_checkpoint = str(tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError, match="SAM checkpoint"):
        build_sam(cfg)
    cfg.model.sam_type = "vit_x"
    with pytest.raises(ValueError, match="sam_type"):
        build_sam(cfg)
    cfg.model.sam_type, cfg.model.out_dim = "vit_h", 128
    with pytest.raises(ValueError, match="out_dim"):
        build_sam(cfg)


def test_encode_sam_checks_the_image_size(tiny):
    model, _ = tiny
    with pytest.raises(ValueError, match="expects images"):
        model.encode_sam(torch.zeros(1, 3, 32, 32))
    embeddings = model.encode_sam(torch.zeros(2, 3, 64, 64))
    assert embeddings.shape == (2, 256, 4, 4)


def test_decoded_mask_has_the_original_resolution_and_depends_on_the_prompt(tiny):
    from cora.model.sam import decode_masks

    model, _ = tiny
    embedding = model.encode_sam(torch.randn(1, 3, 64, 64))
    prompts_ = torch.randn(3, 256)
    masks = decode_masks(model.sam, embedding, prompts_, (48, 64), (90, 120))
    assert masks.shape == (3, 90, 120) and masks.dtype == torch.float32
    assert not torch.allclose(masks[0], masks[1])
    single = decode_masks(model.sam, embedding, prompts_[:1], (48, 64), (90, 120))
    torch.testing.assert_close(single[0], masks[0], atol=1e-5, rtol=1e-4)
