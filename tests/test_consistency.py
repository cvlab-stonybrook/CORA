"""Tests of cora.losses.consistency: Eq. 6 variance and the pixel weights lambda."""
import math

import pytest
import torch

from cora.losses.consistency import confidence_weight, consistency_weight, pixel_variance


def _logits_with_probs(probs):
    """(K-1, H, W) logits whose sigmoid equals ``probs``."""
    probs = torch.as_tensor(probs, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
    return torch.log(probs / (1 - probs)).float()


def test_pixel_variance_is_population_variance():
    x = torch.rand(6, 4, 5)
    ref = ((x - x.mean(0)) ** 2).sum(0) / 6  # divide by K-1 = 6 (number of predictions)
    assert torch.allclose(pixel_variance(x), ref, atol=1e-6)
    assert not torch.allclose(pixel_variance(x), ref * 6 / 5)


def test_pixel_variance_rejects_bad_shape():
    with pytest.raises(ValueError):
        pixel_variance(torch.rand(4, 5))


def test_stable_high_identical_is_one_and_maximally_split_is_zero():
    same = torch.full((6, 3, 3), 1.7)
    assert torch.allclose(consistency_weight(same, "stable_high"), torch.ones(3, 3), atol=1e-6)
    split = torch.cat([torch.full((3, 3, 3), 50.0), torch.full((3, 3, 3), -50.0)])
    assert torch.allclose(consistency_weight(split, "stable_high"), torch.zeros(3, 3), atol=1e-6)
    assert torch.equal(consistency_weight(split, "stable_high", variance_input="binary"), torch.zeros(3, 3))


@pytest.mark.parametrize("variance_input", ["prob", "binary"])
def test_modes_match_formulas(variance_input):
    logits = torch.randn(6, 8, 8) * 2
    values = torch.sigmoid(logits) if variance_input == "prob" else (logits > 0).float()
    v = ((values - values.mean(0)) ** 2).mean(0)
    kw = dict(variance_input=variance_input, exp_scale=7.0)
    assert torch.allclose(consistency_weight(logits, "stable_high", **kw), (1 - v / 0.25).clamp(0, 1), atol=1e-6)
    assert torch.allclose(consistency_weight(logits, "variance", **kw), v, atol=1e-6)
    assert torch.allclose(consistency_weight(logits, "one_minus", **kw), 1 - v, atol=1e-6)
    assert torch.allclose(consistency_weight(logits, "exp", **kw), torch.exp(-7.0 * v), atol=1e-6)


def test_monotonicity_in_variance():
    # one pixel per variance level: k of 6 predictions are 1, the rest 0 -> variance increases up to k = 3
    levels = [0, 1, 2, 3]
    stack = torch.zeros(6, 1, len(levels))
    for col, k in enumerate(levels):
        stack[:k, 0, col] = 1.0
    logits = _logits_with_probs(stack)
    v = pixel_variance(torch.sigmoid(logits))[0]
    assert (v[1:] > v[:-1]).all()
    for mode in ("stable_high", "one_minus", "exp"):
        lam = consistency_weight(logits, mode)[0]
        assert (lam[1:] < lam[:-1]).all(), mode
    lam = consistency_weight(logits, "variance")[0]
    assert (lam[1:] > lam[:-1]).all()


def test_exp_mode_scale():
    logits = torch.randn(6, 4, 4) * 3
    v = pixel_variance(torch.sigmoid(logits))
    assert torch.allclose(consistency_weight(logits, "exp", exp_scale=0.0), torch.ones(4, 4))
    assert torch.allclose(consistency_weight(logits, "exp", exp_scale=20.0), torch.exp(-20.0 * v), atol=1e-6)


def test_result_has_no_gradient_and_is_float32():
    logits = (torch.randn(6, 4, 4) * 2).requires_grad_(True)
    for mode in ("stable_high", "variance", "one_minus", "exp"):
        lam = consistency_weight(logits, mode)
        assert not lam.requires_grad and lam.grad_fn is None and lam.dtype == torch.float32
    lam = consistency_weight(logits.detach().bfloat16(), "stable_high")
    assert lam.dtype == torch.float32


def test_weights_stay_in_unit_interval_for_bounded_modes():
    logits = torch.randn(6, 16, 16) * 5
    for mode in ("stable_high", "one_minus", "exp"):
        lam = consistency_weight(logits, mode)
        assert float(lam.min()) >= 0.0 and float(lam.max()) <= 1.0


def test_full_resolution_runs_in_float32():
    logits = torch.randn(6, 1024, 2048)
    lam = consistency_weight(logits, "stable_high")
    assert lam.shape == (1024, 2048) and math.isfinite(float(lam.mean()))


@pytest.mark.parametrize("kwargs", [{"mode": "bogus"}, {"mode": "exp", "variance_input": "other"},
                                    {"mode": "exp", "exp_scale": -1.0}])
def test_bad_options_raise(kwargs):
    with pytest.raises(ValueError):
        consistency_weight(torch.randn(3, 2, 2), **kwargs)


def test_confidence_weight_threshold_is_inclusive():
    conf = torch.tensor([[0.1, 0.9], [0.95, 0.899]])
    assert torch.equal(confidence_weight(conf, 0.9), torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
