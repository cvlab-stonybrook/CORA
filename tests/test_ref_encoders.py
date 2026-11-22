"""Tests of the anchor mask / box encoders (cora.model.ref_encoders), CPU, tiny LLaVA."""
from __future__ import annotations

import pytest
import torch

from cora.model.llava import clip_cls_features
from cora.model.ref_encoders import AnchorEncoders, place_anchor_embeddings
from tests.fixtures import tiny_model_and_tokenizer


@pytest.fixture(scope="module")
def tiny():
    model, tokenizer = tiny_model_and_tokenizer()
    model.eval()
    return model, tokenizer


def make_encoders(model, **kwargs) -> AnchorEncoders:
    hidden = model.llava.config.text_config.hidden_size
    return AnchorEncoders(hidden, model.llava.model.multi_modal_projector, **kwargs)


def test_clip_cls_is_the_penultimate_layer_cls_token(tiny):
    model, _ = tiny
    pixels = torch.randn(3, 3, 28, 28)
    got = clip_cls_features(model.llava, pixels)
    hidden_states = model.llava.model.vision_tower(pixel_values=pixels, output_hidden_states=True).hidden_states
    torch.testing.assert_close(got, hidden_states[-2][:, 0].float())
    assert got.shape == (3, 32) and not got.requires_grad


def test_separate_projector_starts_as_a_copy_of_llavas_projector(tiny):
    model, _ = tiny
    enc = make_encoders(model, mask_projector="separate")
    llava_projector = model.llava.model.multi_modal_projector
    assert enc.projector is not llava_projector
    for a, b in zip(enc.projector.parameters(), llava_projector.parameters()):
        assert a.data_ptr() != b.data_ptr() and a.requires_grad
        torch.testing.assert_close(a, b)
    cls = torch.randn(4, 32)
    mask_emb, box_emb = enc(cls, torch.rand(4, 4))
    torch.testing.assert_close(mask_emb, llava_projector(cls))
    assert mask_emb.shape == box_emb.shape == (4, 32)
    with torch.no_grad():  # training the copy leaves LLaVA's projector alone
        for p in enc.projector.parameters():
            p.add_(1.0)
    assert not torch.allclose(llava_projector.linear_1.weight, enc.projector.linear_1.weight)


def test_shared_projector_is_llavas_own(tiny):
    model, _ = tiny
    enc = make_encoders(model, mask_projector="shared")
    assert enc.projector is None and enc.active_projector is model.llava.model.multi_modal_projector
    assert {n.split(".")[0] for n, _ in enc.named_parameters()} == {"box_encoder"}
    cls = torch.randn(2, 32)
    mask_emb, _ = enc(cls, torch.rand(2, 4))
    torch.testing.assert_close(mask_emb, model.llava.model.multi_modal_projector(cls))


def test_box_encoder_is_a_linear_map_on_four_numbers(tiny):
    model, _ = tiny
    enc = make_encoders(model)
    boxes = torch.rand(5, 4)
    _, box_emb = enc(torch.randn(5, 32), boxes)
    torch.testing.assert_close(box_emb, enc.box_encoder(boxes))
    assert enc.box_encoder.in_features == 4


def test_shared_projector_trains_only_when_requested(tiny):
    model, _ = tiny
    projector = model.llava.model.multi_modal_projector
    frozen = make_encoders(model, mask_projector="shared", train_projector=False)
    assert len(frozen.own_parameters()) == 2
    assert not projector.linear_1.weight.requires_grad
    trained = make_encoders(model, mask_projector="shared", train_projector=True)
    assert len(trained.own_parameters()) == 2 + 4
    # Regression: the constructor already leaves the shared projector trainable (no set_trainable call needed).
    assert projector.linear_1.weight.requires_grad
    assert all(p.requires_grad for p in trained.own_parameters())
    trained.set_trainable(False)  # restore the shared module for the other tests
    assert not projector.linear_1.weight.requires_grad


def test_set_trainable_and_zero_term(tiny):
    model, _ = tiny
    enc = make_encoders(model)
    term = enc.zero_term()
    assert term.item() == 0.0
    term.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) == 0.0 for p in enc.parameters())
    enc.set_trainable(False)
    assert enc.zero_term() is None
    assert not any(p.requires_grad for p in enc.parameters())


def test_modality_dropout_matches_the_segllm_rule(tiny):
    model, _ = tiny
    enc = make_encoders(model, modality_dropout=0.5)
    cls, boxes = torch.randn(400, 32), torch.rand(400, 4)
    plain = make_encoders(model)
    plain.load_state_dict(enc.state_dict())
    full_mask, full_box = plain(cls, boxes)
    enc.train()
    mask_emb, box_emb = enc(cls, boxes)
    for dropped, full in ((mask_emb, full_mask), (box_emb, full_box)):
        zero = (dropped == 0).all(dim=1)
        assert 0.4 < zero.float().mean() < 0.6  # about half of the references lose this modality
        torch.testing.assert_close(dropped[~zero], full[~zero])  # kept ones are not rescaled
    assert not torch.equal((mask_emb == 0).all(1), (box_emb == 0).all(1))  # independent per modality
    enc.eval()
    mask_emb, box_emb = enc(cls, boxes)
    torch.testing.assert_close(mask_emb, 0.5 * full_mask)
    torch.testing.assert_close(box_emb, 0.5 * full_box)


def test_no_modality_dropout_by_default(tiny):
    model, _ = tiny
    enc = make_encoders(model)
    enc.train()
    cls, boxes = torch.randn(50, 32), torch.rand(50, 4)
    first = enc(cls, boxes)
    second = enc(cls, boxes)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_constructor_validates_options(tiny):
    model, _ = tiny
    with pytest.raises(ValueError, match="mask_projector"):
        make_encoders(model, mask_projector="other")
    with pytest.raises(ValueError, match="modality_dropout"):
        make_encoders(model, modality_dropout=1.0)
    enc = make_encoders(model)
    with pytest.raises(ValueError, match="anchor"):
        enc(torch.randn(2, 32), torch.rand(3, 4))


# ----------------------------------------------------------------------------------------------
# place_anchor_embeddings
# ----------------------------------------------------------------------------------------------
def placeholder_ids(spec):
    mask, box, other = spec.ref_mask_token_id, spec.ref_box_token_id, 7
    return torch.tensor(
        [
            [1, other, mask, box, other, other],
            [1, mask, box, other, mask, box],
            [1, other, other, other, other, other],
        ]
    )


def test_anchor_embeddings_overwrite_only_their_placeholders(tiny):
    model, _ = tiny
    ids = placeholder_ids(model.spec)
    base = torch.randn(3, 6, 8)
    mask_emb, box_emb = torch.randn(3, 8), torch.randn(3, 8)
    ref_seq = torch.tensor([0, 1, 1])
    out = place_anchor_embeddings(base, ids, model.spec, ref_seq, mask_emb, box_emb)
    torch.testing.assert_close(out[0, 2], mask_emb[0])
    torch.testing.assert_close(out[0, 3], box_emb[0])
    torch.testing.assert_close(out[1, 1], mask_emb[1])  # references in placeholder order within a sequence
    torch.testing.assert_close(out[1, 2], box_emb[1])
    torch.testing.assert_close(out[1, 4], mask_emb[2])
    torch.testing.assert_close(out[1, 5], box_emb[2])
    untouched = (ids != model.spec.ref_mask_token_id) & (ids != model.spec.ref_box_token_id)
    torch.testing.assert_close(out[untouched], base[untouched])
    assert out is not base  # out of place: autograd keeps the original embeddings


def test_anchor_embeddings_are_differentiable(tiny):
    model, _ = tiny
    ids = placeholder_ids(model.spec)
    base = torch.randn(3, 6, 8, requires_grad=True)
    mask_emb = torch.randn(3, 8, requires_grad=True)
    box_emb = torch.randn(3, 8, requires_grad=True)
    out = place_anchor_embeddings(base, ids, model.spec, torch.tensor([0, 1, 1]), mask_emb, box_emb)
    out.sum().backward()
    assert mask_emb.grad.abs().sum() > 0 and box_emb.grad.abs().sum() > 0
    assert base.grad[0, 2].abs().sum() == 0  # overwritten positions send no gradient to the token embedding


def test_no_references_is_the_identity(tiny):
    model, _ = tiny
    ids = torch.full((2, 4), 7)
    base = torch.randn(2, 4, 8)
    out = place_anchor_embeddings(base, ids, model.spec, torch.zeros(0, dtype=torch.long), torch.zeros(0, 8), torch.zeros(0, 8))
    assert torch.equal(out, base)


def test_placeholder_reference_mismatches_are_errors(tiny):
    model, _ = tiny
    spec = model.spec
    ids = placeholder_ids(spec)
    base, emb = torch.randn(3, 6, 8), torch.randn(3, 8)
    with pytest.raises(ValueError, match="do not match"):
        place_anchor_embeddings(base, ids, spec, torch.tensor([0, 0, 1]), emb, emb)  # wrong sequence assignment
    with pytest.raises(ValueError, match="sorted"):
        place_anchor_embeddings(base, ids, spec, torch.tensor([1, 1, 0]), emb, emb)
    with pytest.raises(ValueError, match="one mask and one box"):
        place_anchor_embeddings(base, ids, spec, torch.tensor([0, 1, 1]), emb[:2], emb)
    ids_box_missing = ids.clone()
    ids_box_missing[0, 3] = 7
    with pytest.raises(ValueError, match="REF_BOX"):
        place_anchor_embeddings(base, ids_box_missing, spec, torch.tensor([0, 1, 1]), emb, emb)


def test_build_model_leaves_all_anchor_encoder_parameters_trainable_with_a_shared_projector():
    """Regression: with mask_projector='shared' and train_projector=True build_model must not leave LLaVA's
    projector frozen while the box encoder trains (no set_reference_trainable call involved)."""
    from tests.fixtures import tiny_config

    cfg = tiny_config()
    cfg.model.reference.mask_projector = "shared"
    cfg.model.reference.train_projector = True
    model, _ = tiny_model_and_tokenizer(cfg)
    assert all(p.requires_grad for p in model.llava.model.multi_modal_projector.parameters())
    assert all(p.requires_grad for p in model.anchors.own_parameters())
    model.set_reference_trainable(False)
    assert not any(p.requires_grad for p in model.llava.model.multi_modal_projector.parameters())
