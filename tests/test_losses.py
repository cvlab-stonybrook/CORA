"""Tests of cora.losses.seg: LISA losses and the pixel-weighted loss of Eq. 7."""
import math

import pytest
import torch

from cora.losses.seg import dice_loss, sigmoid_ce_loss, weighted_seg_loss


def _data(h=9, w=13, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(h, w, generator=g) * 2).to(dtype)
    target = (torch.rand(h, w, generator=g) > 0.5).to(dtype)
    weight = torch.rand(h, w, generator=g).to(dtype)
    return logits, target, weight


def _ref_bce(logits, target):
    p = torch.sigmoid(logits.double())
    t = target.double()
    return -(t * torch.log(p) + (1 - t) * torch.log(1 - p))


def _ref_dice(p, t, w, scale=1000.0, eps=1e-6):
    num = 2 * (w * p / scale * t).sum()
    den = (w * p / scale).sum() + (w * t / scale).sum()
    return 1 - (num + eps) / (den + eps)


def test_sigmoid_ce_and_dice_match_reference_formulas():
    logits = torch.randn(3, 6, 7)
    target = (torch.rand(3, 6, 7) > 0.5).float()
    bce = sigmoid_ce_loss(logits, target, num_masks=3)
    ref_bce = sum(_ref_bce(logits[i], target[i]).mean() for i in range(3)) / 3
    assert math.isclose(float(bce), float(ref_bce), rel_tol=1e-5)
    dice = dice_loss(logits, target, num_masks=3)
    p = torch.sigmoid(logits.double())
    ref_dice = sum(_ref_dice(p[i], target[i].double(), 1.0) for i in range(3)) / 3
    assert math.isclose(float(dice), float(ref_dice), rel_tol=1e-5)


def test_num_masks_normalisation():
    logits = torch.randn(2, 4, 4)
    target = (torch.rand(2, 4, 4) > 0.5).float()
    assert math.isclose(float(sigmoid_ce_loss(logits, target, 4)) * 2, float(sigmoid_ce_loss(logits, target, 2)),
                        rel_tol=1e-6)
    assert math.isclose(float(dice_loss(logits, target, 4)) * 2, float(dice_loss(logits, target, 2)), rel_tol=1e-6)


@pytest.mark.parametrize("mode", ["plain", "weighted", "weighted_unscaled"])
def test_unit_weights_and_mean_reduction_equal_lisa(mode):
    logits, target, _ = _data()
    bce, dice = weighted_seg_loss(logits, target, torch.ones_like(logits), "mean", mode)
    assert torch.allclose(bce, sigmoid_ce_loss(logits[None], target[None], 1.0), rtol=1e-6)
    assert torch.allclose(dice, dice_loss(logits[None], target[None], 1.0), rtol=1e-6)


def test_off_mode_has_zero_dice_and_keeps_bce():
    logits, target, weight = _data()
    bce, dice = weighted_seg_loss(logits, target, weight, "sum", "off")
    assert float(dice) == 0.0
    assert math.isclose(float(bce), float((weight.double() * _ref_bce(logits, target)).sum()), rel_tol=1e-5)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("mode", ["plain", "weighted", "weighted_unscaled"])
def test_weighted_modes_match_independent_formulas(reduction, mode):
    logits, target, weight = _data(seed=3)
    bce, dice = weighted_seg_loss(logits, target, weight, reduction, mode)
    red = (lambda x: x.sum()) if reduction == "sum" else (lambda x: x.mean())
    w, t, p = weight.double(), target.double(), torch.sigmoid(logits.double())
    ref_bce = red(w * _ref_bce(logits, target))
    ones = torch.ones_like(w)
    ref_d = _ref_dice(p, t, ones if mode == "plain" else w)
    ref_dice = ref_d if mode == "weighted_unscaled" else ref_d * red(w)
    assert math.isclose(float(bce), float(ref_bce), rel_tol=1e-5)
    assert math.isclose(float(dice), float(ref_dice), rel_tol=1e-5)


def test_sum_is_pixel_count_times_mean():
    logits, target, weight = _data()
    n = logits.numel()
    for mode in ("plain", "weighted"):
        b_sum, d_sum = weighted_seg_loss(logits, target, weight, "sum", mode)
        b_mean, d_mean = weighted_seg_loss(logits, target, weight, "mean", mode)
        assert math.isclose(float(b_sum), float(b_mean) * n, rel_tol=1e-5)
        assert math.isclose(float(d_sum), float(d_mean) * n, rel_tol=1e-5)


def test_zero_weights_give_zero_loss():
    logits, target, _ = _data()
    for mode in ("plain", "weighted"):
        bce, dice = weighted_seg_loss(logits, target, torch.zeros_like(logits), "sum", mode)
        assert float(bce) == 0.0 and float(dice) == 0.0


def test_gradient_reaches_logits_but_not_weights():
    logits, target, weight = _data()
    logits.requires_grad_(True)
    weight.requires_grad_(True)
    bce, dice = weighted_seg_loss(logits, target, weight, "sum", "weighted")
    (bce + dice).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert weight.grad is None


def test_float32_safe_at_full_resolution():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(1024, 2048, generator=g) * 3
    target = (torch.rand(1024, 2048, generator=g) > 0.7).float()
    weight = torch.rand(1024, 2048, generator=g)
    bce, dice = weighted_seg_loss(logits, target, weight, "sum", "plain")
    ref_bce = (weight.double() * _ref_bce(logits, target)).sum()
    ref_dice = _ref_dice(torch.sigmoid(logits.double()), target.double(), torch.ones(1, dtype=torch.double)) \
        * weight.double().sum()
    assert math.isclose(float(bce), float(ref_bce), rel_tol=1e-4)
    assert math.isclose(float(dice), float(ref_dice), rel_tol=1e-4)


def test_half_precision_inputs_are_computed_in_float32():
    logits, target, weight = _data()
    bce, dice = weighted_seg_loss(logits.bfloat16(), target.bfloat16(), weight.bfloat16(), "mean", "plain")
    assert bce.dtype == torch.float32 and dice.dtype == torch.float32


@pytest.mark.parametrize("kwargs", [{"reduction": "median"}, {"dice_mode": "bogus"}])
def test_bad_options_raise(kwargs):
    logits, target, weight = _data()
    args = {"reduction": "sum", "dice_mode": "plain", **kwargs}
    with pytest.raises(ValueError):
        weighted_seg_loss(logits, target, weight, **args)


def test_shape_mismatch_raises():
    logits, target, weight = _data()
    with pytest.raises(ValueError):
        weighted_seg_loss(logits, target[:-1], weight, "sum", "plain")
    with pytest.raises(ValueError):
        weighted_seg_loss(logits[None], target[None], weight[None], "sum", "plain")
