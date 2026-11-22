"""Tests of the K-1 branch wrapper (cora.model.teacher) on the tiny model, CPU."""
from __future__ import annotations

import pytest
import torch

from cora.model.teacher import Teacher
from tests.fixtures import make_batch, tiny_model_and_tokenizer


def setup(mode: str = "online", momentum: float = 0.5, seed: int = 0):
    model, tok = tiny_model_and_tokenizer(seed=seed)
    batch = make_batch(model.cfg, tok, model.spec, kind="unlabeled", seqs_per_image=(1, 1), aux_per_seq=3)
    return model, Teacher(model, mode, momentum), batch


def perturb(model, scale: float = 0.2, seed: int = 1) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(scale * torch.randn(p.shape, generator=gen))


def snapshot_of(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def predict(teacher, model, batch):
    with torch.no_grad():
        embeddings = model.encode_sam(batch.aux.images_sam)
    return teacher.predict(batch.aux, embeddings)


def assert_same(a, b, equal: bool = True):
    same = all(torch.allclose(x, y, atol=1e-6, rtol=1e-4) for x, y in zip(a, b))
    assert same == equal


def test_invalid_arguments():
    model, _, _ = setup()
    with pytest.raises(ValueError, match="mode"):
        Teacher(model, "sometimes")
    with pytest.raises(ValueError, match="ema_momentum"):
        Teacher(model, "ema", 1.0)


def test_predict_before_start_is_an_error():
    model, teacher, batch = setup("frozen")
    with pytest.raises(RuntimeError, match="start"):
        predict(teacher, model, batch)


def test_output_shapes_and_no_gradients():
    model, teacher, batch = setup()
    model.train()
    teacher.start()
    logits = predict(teacher, model, batch)
    assert len(logits) == batch.aux.num_sequences == 6
    for n, mask in enumerate(logits):
        assert mask.dtype == torch.float32 and not mask.requires_grad
        assert tuple(mask.shape) == batch.aux.orig_hw[int(batch.aux.seq_image[n])]
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("mode", ["online", "frozen", "ema"])
def test_training_flags_are_restored(mode):
    model, teacher, batch = setup(mode)
    teacher.start()
    for train in (True, False):
        model.train(train)
        flags = {n: m.training for n, m in model.named_modules()}
        predict(teacher, model, batch)
        assert flags == {n: m.training for n, m in model.named_modules()}


def test_online_teacher_equals_the_current_model_in_eval_mode():
    model, teacher, batch = setup("online")
    teacher.start()
    model.train()
    perturb(model)
    got = predict(teacher, model, batch)
    model.eval()
    with torch.no_grad():
        want = model(batch.aux, compute_text_loss=False).mask_logits
    assert_same(got, want)


def test_online_follows_the_model_while_it_trains():
    model, teacher, batch = setup("online")
    teacher.start()
    before = predict(teacher, model, batch)
    perturb(model)
    after = predict(teacher, model, batch)
    assert_same(before, after, equal=False)


def test_frozen_teacher_stays_constant_while_the_model_trains():
    model, teacher, batch = setup("frozen")
    model.eval()
    start = [m.clone() for m in predict(teacher_started(teacher), model, batch)]
    perturb(model)
    model.eval()
    with torch.no_grad():
        live = model(batch.aux, compute_text_loss=False).mask_logits
    assert_same(start, live, equal=False)  # the model moved ...
    assert_same(start, predict(teacher, model, batch))  # ... the teacher did not
    teacher.update()  # no-op for the frozen mode
    assert_same(start, predict(teacher, model, batch))


def teacher_started(teacher):
    teacher.start()
    return teacher


@pytest.mark.parametrize("mode", ["frozen", "ema"])
def test_live_weights_and_optimizer_are_untouched_by_a_teacher_pass(mode):
    model, teacher, batch = setup(mode)
    teacher.start()
    perturb(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    out = model(batch.aux)
    out.ce_loss.backward()
    optimizer.step()
    before = snapshot_of(model)
    ptrs = [p.data_ptr() for p in params]
    state_before = {id(p): {k: v.clone() for k, v in optimizer.state[p].items()} for p in params}
    predict(teacher, model, batch)
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), before[name]), name
    assert [p.data_ptr() for p in params] == ptrs
    for p in params:
        for key, value in optimizer.state[p].items():
            assert torch.equal(value, state_before[id(p)][key])
    optimizer.zero_grad()
    model(batch.aux).ce_loss.backward()  # training continues normally afterwards
    optimizer.step()


def test_live_weights_are_restored_even_when_the_pass_fails():
    model, teacher, batch = setup("frozen")
    teacher.start()
    perturb(model)
    before = snapshot_of(model)
    broken = batch.aux
    broken.input_ids = broken.input_ids.clone()
    broken.input_ids[0][broken.input_ids[0] == model.spec.seg_token_id] = 5  # no <SSEG> -> forward raises
    with pytest.raises(ValueError, match="exactly one"):
        predict(teacher, model, batch)
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), before[name]), name


def test_ema_moves_between_the_snapshot_and_the_live_weights():
    model, teacher, batch = setup("ema", momentum=0.5)
    teacher.start()
    start = predict(teacher, model, batch)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    perturb(model)
    live = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert_same(start, predict(teacher, model, batch))  # nothing moved before update()
    teacher.update()
    for n in names:
        torch.testing.assert_close(teacher._snapshot[n], 0.5 * initial[n] + 0.5 * live[n])
    middle = predict(teacher, model, batch)
    assert_same(start, middle, equal=False)
    model.eval()
    with torch.no_grad():
        current = model(batch.aux, compute_text_loss=False).mask_logits
    assert_same(middle, current, equal=False)
    for _ in range(60):  # converges to the live weights
        teacher.update()
    for n in names:
        torch.testing.assert_close(teacher._snapshot[n], live[n], atol=1e-6, rtol=1e-6)
    for got, want in zip(predict(teacher, model, batch), current):
        torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-3)


def test_update_is_a_no_op_for_online_and_frozen():
    for mode in ("online", "frozen"):
        model, teacher, batch = setup(mode)
        teacher.start()
        before = {n: t.clone() for n, t in teacher._snapshot.items()}
        perturb(model)
        teacher.update()
        assert all(torch.equal(before[n], teacher._snapshot[n]) for n in before)


def test_state_dict_round_trip_for_resume(tmp_path):
    model, teacher, batch = setup("frozen")
    teacher.start()
    expected = predict(teacher, model, batch)
    state = teacher.state_dict()
    torch.save(state, tmp_path / "teacher.pt")
    loaded = torch.load(tmp_path / "teacher.pt", map_location="cpu", weights_only=True)
    perturb(model)
    resumed = Teacher(model, "frozen")
    resumed.load_state_dict(loaded)
    assert_same(expected, predict(resumed, model, batch))
    with pytest.raises(ValueError, match="teacher"):
        Teacher(model, "ema").load_state_dict(loaded)
    bad = dict(loaded)
    bad["snapshot"] = {"unknown.weight": torch.zeros(1)}
    with pytest.raises(KeyError):
        Teacher(model, "frozen").load_state_dict(bad)


def test_online_state_dict_resume():
    model, teacher, batch = setup("online")
    teacher.start()
    other = Teacher(model, "online")
    other.load_state_dict(teacher.state_dict())
    assert_same(predict(other, model, batch), predict(teacher, model, batch))


def test_reference_encoders_in_the_snapshot():
    """Trainable anchor encoders are part of the snapshot, so a frozen teacher is faithful for conditional queries."""
    model, teacher, _ = setup("frozen")
    teacher.start()
    assert any(n.startswith("anchors") for n in teacher._snapshot)
    assert any("lora_" in n for n in teacher._snapshot) and "llava.lm_head.weight" in teacher._snapshot
    assert all(not t.requires_grad for t in teacher._snapshot.values())
