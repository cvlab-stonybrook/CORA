"""Tests of cora.losses.criterion: labeled loss, pixel weights and unlabeled loss (Eq. 4-9)."""
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cora.config import LossConfig, TFCAConfig, UnlabeledConfig
from cora.data.batch import Batch
from cora.losses.consistency import consistency_weight
from cora.losses.criterion import effective_dice_mode, labeled_loss, pixel_weights, unlabeled_loss
from cora.losses.seg import dice_loss, sigmoid_ce_loss, weighted_seg_loss
from cora.losses.token_bank import TokenBank


def _f(x):
    return float(x.detach())

HW = (6, 8)


def _batch(masks, seq_image=None, class_ids=None, aux_group=None, confidence=None, kind="labeled"):
    n = len(masks)
    seq_image = torch.tensor(seq_image if seq_image is not None else list(range(n)))
    aux = None
    if aux_group is not None:
        m = len(aux_group)
        aux = _batch([torch.zeros(HW, dtype=torch.uint8)] * m, kind="aux")
        aux.group = torch.tensor(aux_group)
    return Batch(
        kind=kind, image_ids=[f"img{i}" for i in range(int(seq_image.max()) + 1)],
        images_sam=torch.zeros(1, 3, 4, 4), images_clip=torch.zeros(1, 3, 4, 4), resize_hw=[HW], orig_hw=[HW],
        input_ids=torch.zeros(n, 3, dtype=torch.long), attention_mask=torch.ones(n, 3, dtype=torch.long),
        labels=torch.zeros(n, 3, dtype=torch.long), seq_image=seq_image, masks=list(masks),
        class_ids=torch.tensor(class_ids if class_ids is not None else [0] * n),
        set_names=["x"] * n, ref_pixels=torch.zeros(0, 3, 4, 4), ref_boxes=torch.zeros(0, 4),
        ref_seq=torch.zeros(0, dtype=torch.long), aux=aux, confidence=confidence)


def _masks(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.rand(*HW, generator=g) > 0.5).to(torch.uint8) for _ in range(n)]


def _out(n, dim=4, seed=1, ce=None, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        mask_logits=[(torch.randn(*HW, generator=g) * 2).requires_grad_(requires_grad) for _ in range(n)],
        seg_embed=torch.randn(n, dim, generator=g).requires_grad_(requires_grad),
        seg_hidden=torch.randn(n, dim + 3, generator=g).requires_grad_(requires_grad), ce_loss=ce)


# ------------------------------------------------------------------------------------------ labeled loss
def test_labeled_loss_reproduces_lisa_assembly_with_multiple_masks_per_image():
    masks = _masks(5)
    masks[0][0, 0] = 255  # ignored pixel counts as background
    seq_image = [0, 0, 0, 1, 1]
    out = _out(5, ce=torch.tensor(1.25))
    batch = _batch(masks, seq_image=seq_image)
    lcfg = LossConfig(ce_weight=0.7, bce_weight=2.0, dice_weight=0.5)
    res = labeled_loss(out, batch, lcfg)

    gts = [(m == 1).float() for m in masks]
    ref_bce = ref_dice = 0.0
    for idx in ([0, 1, 2], [3, 4]):
        pred = torch.stack([out.mask_logits[i].detach() for i in idx])
        gt = torch.stack([gts[i] for i in idx])
        ref_bce += float(sigmoid_ce_loss(pred, gt, len(idx))) * len(idx)
        ref_dice += float(dice_loss(pred, gt, len(idx))) * len(idx)
    ref_bce, ref_dice = ref_bce / 5, ref_dice / 5
    assert math.isclose(_f(res["bce"]), ref_bce, rel_tol=1e-5)
    assert math.isclose(_f(res["dice"]), ref_dice, rel_tol=1e-5)
    assert _f(res["ce"]) == 1.25
    assert math.isclose(_f(res["loss"]), 0.7 * 1.25 + 2.0 * ref_bce + 0.5 * ref_dice, rel_tol=1e-5)
    # the global per-mask mean equals the plain mean of the per-mask values
    per_mask = [float(sigmoid_ce_loss(out.mask_logits[i][None].detach(), gts[i][None], 1.0)) for i in range(5)]
    assert math.isclose(_f(res["bce"]), sum(per_mask) / 5, rel_tol=1e-5)


def test_labeled_loss_defaults_are_unit_weights_and_backprop():
    out = _out(2, ce=torch.tensor(0.5, requires_grad=True))
    res = labeled_loss(out, _batch(_masks(2)), LossConfig())
    assert math.isclose(_f(res["loss"]), 0.5 + _f(res["bce"]) + _f(res["dice"]), rel_tol=1e-6)
    res["loss"].backward()
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in out.mask_logits)
    assert out.ce_loss.grad == 1.0


def test_labeled_loss_without_ce():
    out = _out(2, ce=None)
    with pytest.raises(ValueError):
        labeled_loss(out, _batch(_masks(2)), LossConfig())
    res = labeled_loss(out, _batch(_masks(2)), LossConfig(ce_weight=0.0))
    assert _f(res["ce"]) == 0.0


def test_labeled_loss_input_checks():
    with pytest.raises(ValueError):
        labeled_loss(_out(2, ce=torch.tensor(0.0)), _batch(_masks(3)), LossConfig())
    out = _out(1, ce=torch.tensor(0.0))
    out.mask_logits[0] = torch.zeros(3, 3)
    with pytest.raises(ValueError):
        labeled_loss(out, _batch(_masks(1)), LossConfig())


# ------------------------------------------------------------------------------------------ pixel weights
def _logit_stack(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*HW, generator=g) * 2 for _ in range(4)]


def test_pixel_weights_group_teacher_logits_by_aux_group():
    # aux sequences are interleaved: group says which trained sequence each belongs to
    group = [1, 0, 1, 0, 1, 0]
    teacher = _logit_stack(0) + _logit_stack(1)[:2]
    batch = _batch(_masks(2), aux_group=group, kind="unlabeled")
    for mode in ("stable_high", "variance", "one_minus", "exp"):
        ucfg = UnlabeledConfig(weight_mode=mode, variance_input="binary")
        weights = pixel_weights(batch, teacher, ucfg)
        assert len(weights) == 2
        for i in range(2):
            members = [teacher[j] for j in range(6) if group[j] == i]
            ref = consistency_weight(torch.stack(members), mode, "binary", ucfg.exp_scale)
            assert torch.allclose(weights[i], ref)
            assert weights[i].shape == HW and not weights[i].requires_grad


def test_pixel_weights_no_gradient_through_teacher_logits():
    teacher = [t.clone().requires_grad_(True) for t in _logit_stack(2)]
    batch = _batch(_masks(1), aux_group=[0, 0, 0, 0], kind="unlabeled")
    weights = pixel_weights(batch, teacher, UnlabeledConfig())
    assert weights[0].grad_fn is None and not weights[0].requires_grad


def test_pixel_weights_confidence_and_none():
    conf = [torch.rand(*HW), torch.rand(*HW)]
    batch = _batch(_masks(2), confidence=conf, kind="unlabeled")
    ucfg = UnlabeledConfig(weighting="confidence", confidence_threshold=0.5)
    weights = pixel_weights(batch, None, ucfg)
    for w, c in zip(weights, conf):
        assert torch.equal(w, (c >= 0.5).float())
    ones = pixel_weights(batch, None, UnlabeledConfig(weighting="none"))
    assert all(torch.equal(w, torch.ones(HW)) for w in ones)
    with pytest.raises(ValueError):
        pixel_weights(_batch(_masks(2), kind="unlabeled"), None, ucfg)


def test_pixel_weights_errors():
    batch = _batch(_masks(2), aux_group=[0, 0], kind="unlabeled")
    with pytest.raises(ValueError):  # trained sequence 1 has no rephrasings
        pixel_weights(batch, _logit_stack(0)[:2], UnlabeledConfig())
    for bad_group in ([0, 2], [-1, 1]):  # aux group ids outside [0, N) must not be dropped silently
        bad = _batch(_masks(2), aux_group=bad_group, kind="unlabeled")
        with pytest.raises(ValueError, match="group"):
            pixel_weights(bad, _logit_stack(0)[:2], UnlabeledConfig())
    with pytest.raises(ValueError):  # wrong number of teacher logits
        pixel_weights(batch, _logit_stack(0)[:1], UnlabeledConfig())
    with pytest.raises(ValueError):  # no aux batch
        pixel_weights(_batch(_masks(1), kind="unlabeled"), _logit_stack(0), UnlabeledConfig())
    with pytest.raises(ValueError):
        pixel_weights(batch, None, UnlabeledConfig())
    with pytest.raises(ValueError):
        pixel_weights(batch, _logit_stack(0)[:2], UnlabeledConfig(weighting="bogus"))
    with pytest.raises(ValueError):  # shape mismatch
        pixel_weights(_batch(_masks(1), aux_group=[0, 0], kind="unlabeled"),
                      [torch.zeros(2, 2), torch.zeros(2, 2)], UnlabeledConfig())


# ------------------------------------------------------------------------------------------ unlabeled loss
def _weights(n, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(*HW, generator=g) for _ in range(n)]


def _filled_bank(dim=4, size=8):
    g = torch.Generator().manual_seed(9)
    bank = TokenBank(3, dim, size)
    bank.enqueue(torch.randn(9, dim, generator=g), torch.tensor([0, 1, 2] * 3))
    return bank


def test_unlabeled_loss_is_sigma_times_sum_of_eq7_and_eq8():
    n = 3
    out = _out(n)
    batch = _batch(_masks(n), class_ids=[0, 1, 2], kind="unlabeled")
    weights = _weights(n)
    bank = _filled_bank()
    ucfg = UnlabeledConfig(weight=0.01, bce_weight=1.0, dice_weight=1.0)
    tcfg = TFCAConfig(num_negatives=0, num_positives=0, temperature=0.1)
    res = unlabeled_loss(out, batch, weights, bank, ucfg, tcfg)

    parts = [weighted_seg_loss(out.mask_logits[i].detach(), (batch.masks[i] == 1).float(), weights[i], "sum", "plain")
             for i in range(n)]
    ref_bce = sum(float(p[0]) for p in parts)
    ref_dice = sum(float(p[1]) for p in parts)
    assert math.isclose(_f(res["bce"]), ref_bce, rel_tol=1e-5)
    assert math.isclose(_f(res["dice"]), ref_dice, rel_tol=1e-5)
    assert math.isclose(_f(res["seg"]), ref_bce + ref_dice, rel_tol=1e-5)
    assert _f(res["pairs"]) > 0 and _f(res["tfca"]) > 0
    assert math.isclose(_f(res["loss"]), 0.01 * (_f(res["seg"]) + _f(res["tfca"])), rel_tol=1e-5)
    assert math.isclose(_f(res["weight_mean"]), float(torch.stack([w.mean() for w in weights]).mean()),
                        rel_tol=1e-6)


def test_sigma_scales_both_terms_linearly():
    out = _out(2)
    batch = _batch(_masks(2), class_ids=[0, 1], kind="unlabeled")
    bank, tcfg = _filled_bank(), TFCAConfig(num_negatives=0, num_positives=0)
    a = unlabeled_loss(out, batch, _weights(2), bank, UnlabeledConfig(weight=0.001), tcfg)
    b = unlabeled_loss(out, batch, _weights(2), bank, UnlabeledConfig(weight=0.1), tcfg)
    assert math.isclose(_f(b["loss"]), 100 * _f(a["loss"]), rel_tol=1e-5)
    assert _f(a["seg"]) == _f(b["seg"]) and _f(a["tfca"]) == _f(b["tfca"])
    zero = unlabeled_loss(out, batch, _weights(2), bank, UnlabeledConfig(weight=0.0), tcfg)
    assert _f(zero["loss"]) == 0.0 and _f(zero["seg"]) > 0


def test_unlabeled_loss_bce_dice_weights_reduction_and_dice_mode():
    out = _out(1)
    batch = _batch(_masks(1), kind="unlabeled")
    w = _weights(1)
    ucfg = UnlabeledConfig(bce_weight=2.0, dice_weight=0.5, reduction="mean", dice_mode="weighted", weight=1.0)
    res = unlabeled_loss(out, batch, w, None, ucfg, TFCAConfig(enabled=False))
    b, d = weighted_seg_loss(out.mask_logits[0].detach(), (batch.masks[0] == 1).float(), w[0], "mean", "weighted")
    assert math.isclose(_f(res["seg"]), 2.0 * float(b) + 0.5 * float(d), rel_tol=1e-5)
    off = unlabeled_loss(out, batch, w, None, replace(ucfg, dice_mode="off"), TFCAConfig(enabled=False))
    assert _f(off["dice"]) == 0.0


def test_effective_dice_mode_only_changes_plain_with_confidence_weighting():
    assert effective_dice_mode(UnlabeledConfig(weighting="confidence", dice_mode="plain")) == "weighted"
    for mode in ("weighted", "weighted_unscaled", "off"):
        assert effective_dice_mode(UnlabeledConfig(weighting="confidence", dice_mode=mode)) == mode
    for weighting in ("consistency", "none"):
        assert effective_dice_mode(UnlabeledConfig(weighting=weighting, dice_mode="plain")) == "plain"


@pytest.mark.parametrize("dice_mode", ["plain", "weighted", "weighted_unscaled"])
def test_confidence_baseline_below_threshold_pixels_leave_bce_and_dice(dice_mode):
    """With OCPL off, pixels under the threshold supervise neither BCE nor Dice (default dice_mode too)."""
    g = torch.Generator().manual_seed(3)
    conf = torch.rand(*HW, generator=g)
    conf[:, : HW[1] // 2] = 0.1  # left half below the 0.9 threshold
    conf[:, HW[1] // 2:] = 0.95
    ucfg = UnlabeledConfig(weighting="confidence", confidence_threshold=0.9, dice_mode=dice_mode, weight=1.0)
    batch = _batch(_masks(1), confidence=[conf], kind="unlabeled")
    weights = pixel_weights(batch, None, ucfg)
    out = _out(1)
    base = unlabeled_loss(out, batch, weights, None, ucfg, TFCAConfig(enabled=False))

    garbage = _out(1, seed=7)
    logits = out.mask_logits[0].detach().clone()
    logits[:, : HW[1] // 2] = garbage.mask_logits[0].detach()[:, : HW[1] // 2] * 5
    masks = batch.masks[0].clone()
    masks[:, : HW[1] // 2] = 1 - masks[:, : HW[1] // 2]
    changed_out = SimpleNamespace(mask_logits=[logits], seg_embed=out.seg_embed, seg_hidden=out.seg_hidden,
                                  ce_loss=None)
    changed_batch = _batch([masks], confidence=[conf], kind="unlabeled")
    changed = unlabeled_loss(changed_out, changed_batch, weights, None, ucfg, TFCAConfig(enabled=False))
    assert math.isclose(_f(base["bce"]), _f(changed["bce"]), rel_tol=1e-5)
    assert math.isclose(_f(base["dice"]), _f(changed["dice"]), rel_tol=1e-5)


def test_consistency_weighting_keeps_the_literal_plain_dice():
    out = _out(1)
    batch = _batch(_masks(1), kind="unlabeled")
    w = _weights(1)
    ucfg = UnlabeledConfig(weighting="consistency", dice_mode="plain", weight=1.0)
    res = unlabeled_loss(out, batch, w, None, ucfg, TFCAConfig(enabled=False))
    _, d = weighted_seg_loss(out.mask_logits[0].detach(), (batch.masks[0] == 1).float(), w[0], ucfg.reduction,
                             "plain")
    assert math.isclose(_f(res["dice"]), float(d), rel_tol=1e-5)


@pytest.mark.parametrize("feature,attr", [("projected", "seg_embed"), ("hidden", "seg_hidden")])
def test_tfca_feature_selection_and_gradient_paths(feature, attr):
    out = _out(2, dim=4)
    batch = _batch(_masks(2), class_ids=[0, 1], kind="unlabeled")
    dim = getattr(out, attr).shape[1]
    bank = TokenBank(3, dim, 8)
    bank.enqueue(torch.randn(6, dim), torch.tensor([0, 0, 1, 1, 2, 2]))
    tcfg = TFCAConfig(feature=feature, num_negatives=0, num_positives=0)
    res = unlabeled_loss(out, batch, _weights(2), bank, UnlabeledConfig(weight=1.0, dice_mode="off"), tcfg)
    res["tfca"].backward()
    assert getattr(out, attr).grad.abs().sum() > 0
    other = out.seg_hidden if attr == "seg_embed" else out.seg_embed
    assert other.grad is None
    assert all(t.grad is None for t in out.mask_logits)  # Eq. 8 does not touch the mask logits


def test_seg_gradient_reaches_logits_and_not_weights():
    out = _out(1)
    w = _weights(1)[0].requires_grad_(True)
    res = unlabeled_loss(out, _batch(_masks(1), kind="unlabeled"), [w], None, UnlabeledConfig(), TFCAConfig(enabled=False))
    res["loss"].backward()
    assert out.mask_logits[0].grad.abs().sum() > 0 and w.grad is None
    assert out.seg_embed.grad is None


def test_empty_bank_and_no_valid_anchor_give_zero_tfca_with_graph():
    out = _out(2)
    batch = _batch(_masks(2), class_ids=[0, 1], kind="unlabeled")
    ucfg, tcfg = UnlabeledConfig(weight=1.0), TFCAConfig()
    res = unlabeled_loss(out, batch, _weights(2), TokenBank(3, 4, 8), ucfg, tcfg)
    assert _f(res["tfca"]) == 0.0 and _f(res["pairs"]) == 0.0
    assert math.isclose(_f(res["loss"]), _f(res["seg"]), rel_tol=1e-6)
    res["loss"].backward()
    # anchors with unknown class
    batch = _batch(_masks(2), class_ids=[-1, -1], kind="unlabeled")
    res = unlabeled_loss(_out(2), batch, _weights(2), _filled_bank(), ucfg, tcfg)
    assert _f(res["tfca"]) == 0.0 and _f(res["pairs"]) == 0.0


def test_tfca_disabled_gives_zero_term_with_or_without_bank():
    out = _out(2)
    batch = _batch(_masks(2), class_ids=[0, 1], kind="unlabeled")
    ucfg = UnlabeledConfig(weight=1.0)
    for bank in (_filled_bank(), None):
        res = unlabeled_loss(out, batch, _weights(2), bank, ucfg, TFCAConfig(enabled=False))
        assert _f(res["tfca"]) == 0.0 and _f(res["pairs"]) == 0.0
        assert math.isclose(_f(res["loss"]), _f(res["seg"]), rel_tol=1e-6)


def test_tfca_enabled_without_bank_raises_instead_of_dropping_eq8():
    batch = _batch(_masks(2), class_ids=[0, 1], kind="unlabeled")
    with pytest.raises(ValueError, match="TokenBank"):
        unlabeled_loss(_out(2), batch, _weights(2), None, UnlabeledConfig(), TFCAConfig(enabled=True))


def test_unlabeled_loss_input_checks():
    out = _out(2)
    batch = _batch(_masks(2), kind="unlabeled")
    with pytest.raises(ValueError):
        unlabeled_loss(out, batch, _weights(1), None, UnlabeledConfig(), TFCAConfig())
    with pytest.raises(ValueError):
        unlabeled_loss(out, batch, [torch.ones(2, 2)] * 2, None, UnlabeledConfig(), TFCAConfig())
    with pytest.raises(ValueError):
        unlabeled_loss(out, batch, _weights(2), _filled_bank(), UnlabeledConfig(), TFCAConfig(feature="bogus"))


def test_full_resolution_unlabeled_step_is_finite():
    h, w = 1024, 2048
    g = torch.Generator().manual_seed(0)
    mask = (torch.rand(h, w, generator=g) > 0.6).to(torch.uint8)
    out = SimpleNamespace(mask_logits=[(torch.randn(h, w, generator=g) * 2).requires_grad_(True)],
                          seg_embed=torch.randn(1, 4, requires_grad=True), seg_hidden=torch.randn(1, 4),
                          ce_loss=None)
    batch = _batch([mask], kind="unlabeled")
    teacher = [torch.randn(h, w, generator=g) for _ in range(6)]
    batch.aux = _batch([mask] * 6, kind="aux")
    batch.aux.group = torch.zeros(6, dtype=torch.long)
    weights = pixel_weights(batch, teacher, UnlabeledConfig())
    res = unlabeled_loss(out, batch, weights, None, UnlabeledConfig(), TFCAConfig(enabled=False))
    res["loss"].backward()
    assert torch.isfinite(res["loss"]) and torch.isfinite(out.mask_logits[0].grad).all()


@pytest.mark.gpu
def test_unlabeled_loss_on_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    out = _out(2)
    batch = _batch(_masks(2), class_ids=[0, 1], kind="unlabeled")
    weights, bank = _weights(2), _filled_bank()
    ucfg, tcfg = UnlabeledConfig(weight=1.0), TFCAConfig(num_negatives=0, num_positives=0)
    cpu = unlabeled_loss(out, batch, weights, bank, ucfg, tcfg)
    out_gpu = SimpleNamespace(mask_logits=[t.detach().cuda() for t in out.mask_logits],
                              seg_embed=out.seg_embed.detach().cuda(), seg_hidden=out.seg_hidden.detach().cuda(),
                              ce_loss=None)
    gpu = unlabeled_loss(out_gpu, batch, [w.cuda() for w in weights], bank.cuda(), ucfg, tcfg)
    for key in ("seg", "tfca", "loss"):
        assert math.isclose(_f(cpu[key]), _f(gpu[key]), rel_tol=1e-4)
