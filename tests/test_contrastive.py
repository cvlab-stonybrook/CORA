"""Tests of cora.losses.contrastive (Eq. 8)."""
import itertools
import math

import pytest
import torch

from cora.losses.contrastive import token_contrastive_loss


def _cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def test_hand_computed_example():
    tau = 0.5
    anchors = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[1.0, 1.0], [0.0, 2.0], [-1.0, 0.0]])
    classes = torch.tensor([0, 1, 1])
    s_pos = 1 / math.sqrt(2)
    s_neg = [0.0, -1.0]
    expected = -math.log(math.exp(s_pos / tau) / (math.exp(s_pos / tau) + sum(math.exp(s / tau) for s in s_neg)))
    loss, pairs = token_contrastive_loss(anchors, torch.tensor([0]), bank, classes, tau, num_positives=0,
                                         num_negatives=0)
    assert pairs == 1 and math.isclose(float(loss), expected, rel_tol=1e-6)


def test_matches_independent_loop_for_several_anchors_positives_and_reductions():
    g = torch.Generator().manual_seed(0)
    anchors = torch.randn(5, 8, generator=g)
    a_cls = torch.tensor([0, 1, 2, 0, 1])
    bank = torch.randn(20, 8, generator=g)
    b_cls = torch.arange(20) % 3
    tau = 0.1
    terms = []
    for a, c in zip(anchors, a_cls.tolist()):
        pos = [bank[j] for j in range(20) if b_cls[j] == c]
        neg = [bank[j] for j in range(20) if b_cls[j] != c]
        for p in pos:
            den = math.exp(_cos(a, p) / tau) + sum(math.exp(_cos(a, n) / tau) for n in neg)
            terms.append(-math.log(math.exp(_cos(a, p) / tau) / den))
    for reduction, ref in (("sum", sum(terms)), ("mean", sum(terms) / len(terms))):
        loss, pairs = token_contrastive_loss(anchors, a_cls, bank, b_cls, tau, 0, 0, reduction)
        assert pairs == len(terms) and math.isclose(float(loss), ref, rel_tol=1e-4)


def test_gradient_reaches_anchors_but_not_the_bank():
    anchors = torch.randn(3, 6, requires_grad=True)
    bank = torch.randn(10, 6, requires_grad=True)
    loss, _ = token_contrastive_loss(anchors, torch.tensor([0, 1, 0]), bank, torch.arange(10) % 2, 0.07)
    loss.backward()
    assert anchors.grad is not None and anchors.grad.abs().sum() > 0
    assert bank.grad is None


def test_loss_decreases_when_anchor_moves_to_positive():
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    classes = torch.tensor([0, 1])
    near, _ = token_contrastive_loss(torch.tensor([[1.0, 0.1]]), torch.tensor([0]), bank, classes, 0.1, 0, 0)
    far, _ = token_contrastive_loss(torch.tensor([[0.1, 1.0]]), torch.tensor([0]), bank, classes, 0.1, 0, 0)
    assert near < far


def test_scale_invariance_of_cosine_similarity():
    anchors = torch.randn(2, 5)
    bank = torch.randn(8, 5)
    a_cls, b_cls = torch.tensor([0, 1]), torch.arange(8) % 2
    ref, _ = token_contrastive_loss(anchors, a_cls, bank, b_cls, 0.2, 0, 0)
    scaled, _ = token_contrastive_loss(anchors * 7.0, a_cls, bank * 0.3, b_cls, 0.2, 0, 0)
    assert torch.allclose(ref, scaled, atol=1e-5)


def test_sampling_counts_and_reproducibility():
    anchors = torch.randn(1, 4)
    bank = torch.randn(40, 4)
    b_cls = torch.arange(40) % 2
    a_cls = torch.tensor([0])
    _, pairs = token_contrastive_loss(anchors, a_cls, bank, b_cls, 0.1, num_positives=3, num_negatives=5)
    assert pairs == 3
    _, pairs = token_contrastive_loss(anchors, a_cls, bank, b_cls, 0.1, num_positives=0, num_negatives=5)
    assert pairs == 20
    # fewer entries than requested -> use all that exist
    _, pairs = token_contrastive_loss(anchors, a_cls, bank, b_cls, 0.1, num_positives=500, num_negatives=500)
    assert pairs == 20
    a = token_contrastive_loss(anchors, a_cls, bank, b_cls, 0.1, 1, 4, generator=torch.Generator().manual_seed(5))[0]
    b = token_contrastive_loss(anchors, a_cls, bank, b_cls, 0.1, 1, 4, generator=torch.Generator().manual_seed(5))[0]
    assert torch.equal(a, b)
    others = {float(token_contrastive_loss(anchors, a_cls, bank, b_cls, 0.1, 1, 4,
                                           generator=torch.Generator().manual_seed(s))[0]) for s in range(8)}
    assert len(others) > 1


def _loss_with_negatives(anchor, positive, negatives, tau):
    """Independent single-pair Eq. 8 value for an explicit list of negatives."""
    den = math.exp(_cos(anchor, positive) / tau) + sum(math.exp(_cos(anchor, n) / tau) for n in negatives)
    return -math.log(math.exp(_cos(anchor, positive) / tau) / den)


def test_sampled_negatives_are_distinct_other_class_subsets_of_the_requested_size():
    tau = 0.5
    g = torch.Generator().manual_seed(3)
    anchor = torch.randn(1, 4, generator=g)
    bank = torch.randn(5, 4, generator=g)
    b_cls = torch.tensor([0, 1, 1, 2, 2])  # one positive, four negatives
    negs = [bank[j] for j in (1, 2, 3, 4)]
    subsets = {c: _loss_with_negatives(anchor[0], bank[0], [negs[k] for k in c], tau)
               for c in itertools.combinations(range(4), 2)}
    seen = set()
    for seed in range(200):
        loss, pairs = token_contrastive_loss(anchor, torch.tensor([0]), bank, b_cls, tau, 1, 2,
                                             generator=torch.Generator().manual_seed(seed))
        assert pairs == 1
        matches = [c for c, ref in subsets.items() if math.isclose(float(loss), ref, rel_tol=1e-5)]
        assert len(matches) == 1, "sampled loss is not the loss of any 2-subset of distinct other-class entries"
        seen.add(matches[0])
    assert seen == set(subsets)  # every 2-subset occurs; with-replacement draws (a, a) would match none


def test_requesting_more_negatives_than_exist_uses_all_and_exactly_n_are_drawn_otherwise():
    g = torch.Generator().manual_seed(4)
    anchor = torch.randn(1, 4, generator=g)
    bank = torch.randn(6, 4, generator=g)
    b_cls = torch.tensor([0, 1, 1, 1, 2, 2])
    a_cls = torch.tensor([0])
    full = token_contrastive_loss(anchor, a_cls, bank, b_cls, 0.3, 1, 0)[0]
    for wanted in (5, 6, 500):  # 5 negatives exist
        assert torch.allclose(token_contrastive_loss(anchor, a_cls, bank, b_cls, 0.3, 1, wanted)[0], full)
    # wanted == 4 of 5 must drop exactly one negative: the loss equals one of the five leave-one-out values
    loo = [_loss_with_negatives(anchor[0], bank[0], [bank[j] for j in (1, 2, 3, 4, 5) if j != d], 0.3)
           for d in (1, 2, 3, 4, 5)]
    for seed in range(20):
        v = float(token_contrastive_loss(anchor, a_cls, bank, b_cls, 0.3, 1, 4,
                                         generator=torch.Generator().manual_seed(seed))[0])
        assert any(math.isclose(v, ref, rel_tol=1e-5) for ref in loo)
        assert not math.isclose(v, float(full), rel_tol=1e-6)


def test_sampled_positives_are_distinct_and_from_the_anchor_class():
    g = torch.Generator().manual_seed(6)
    anchor = torch.randn(1, 4, generator=g)
    bank = torch.randn(5, 4, generator=g)
    b_cls = torch.tensor([0, 0, 0, 1, 1])
    negs = [bank[3], bank[4]]
    per_positive = [_loss_with_negatives(anchor[0], bank[k], negs, 0.4) for k in range(3)]
    seen = set()
    for seed in range(60):
        loss, pairs = token_contrastive_loss(anchor, torch.tensor([0]), bank, b_cls, 0.4, 2, 0, "sum",
                                             generator=torch.Generator().manual_seed(seed))
        assert pairs == 2
        sums = {pair: per_positive[pair[0]] + per_positive[pair[1]] for pair in itertools.combinations(range(3), 2)}
        match = [pair for pair, ref in sums.items() if math.isclose(float(loss), ref, rel_tol=1e-5)]
        assert len(match) == 1  # two distinct positives of class 0 (a repeated positive would match none)
        seen.add(match[0])
    assert len(seen) == 3


def test_anchor_without_positive_or_negative_is_skipped():
    anchors = torch.randn(3, 4, requires_grad=True)
    bank = torch.randn(4, 4)
    b_cls = torch.tensor([0, 0, 1, 1])
    # class 2 has no positive; class 0 anchor is valid; class -1 is unknown
    loss, pairs = token_contrastive_loss(anchors, torch.tensor([0, 2, -1]), bank, b_cls, 0.1, 1, 0)
    assert pairs == 1
    assert loss.requires_grad
    # bank holds only one class -> no negatives -> nothing valid
    loss, pairs = token_contrastive_loss(anchors, torch.tensor([0, 0, 0]), bank[:2], b_cls[:2], 0.1)
    assert pairs == 0 and float(loss.detach()) == 0.0


def test_no_valid_anchor_returns_zero_that_depends_on_anchors():
    anchors = torch.randn(2, 4, requires_grad=True)
    for bank, b_cls in ((torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)),
                        (torch.randn(3, 4), torch.zeros(3, dtype=torch.long))):
        loss, pairs = token_contrastive_loss(anchors, torch.tensor([0, 1]), bank, b_cls, 0.07)
        assert pairs == 0 and float(loss.detach()) == 0.0 and loss.requires_grad
        loss.backward()
        assert torch.equal(anchors.grad, torch.zeros_like(anchors))
        anchors.grad = None
    loss, pairs = token_contrastive_loss(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), torch.randn(3, 4),
                                         torch.zeros(3, dtype=torch.long), 0.07)
    assert pairs == 0 and float(loss) == 0.0


def test_numerically_stable_with_tiny_temperature():
    anchors = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    loss, _ = token_contrastive_loss(anchors, torch.tensor([0]), bank, torch.tensor([0, 1]), 1e-3, 0, 0)
    assert math.isfinite(float(loss)) and float(loss) < 1e-6
    loss, _ = token_contrastive_loss(-anchors, torch.tensor([0]), bank, torch.tensor([0, 1]), 1e-3, 0, 0)
    assert math.isclose(float(loss), 2000.0, rel_tol=1e-4)


def test_bank_entries_with_negative_class_are_never_used():
    anchors = torch.randn(1, 4)
    bank = torch.randn(4, 4)
    with_invalid = token_contrastive_loss(anchors, torch.tensor([0]), bank, torch.tensor([0, 1, -1, -1]), 0.1, 0, 0)
    without = token_contrastive_loss(anchors, torch.tensor([0]), bank[:2], torch.tensor([0, 1]), 0.1, 0, 0)
    assert torch.allclose(with_invalid[0], without[0])


@pytest.mark.parametrize("kwargs", [{"temperature": 0.0}, {"reduction": "max"}, {"num_positives": -1}])
def test_bad_options_raise(kwargs):
    args = dict(anchors=torch.randn(1, 4), anchor_classes=torch.tensor([0]), bank_features=torch.randn(2, 4),
                bank_classes=torch.tensor([0, 1]), temperature=0.1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        token_contrastive_loss(**args)


def test_shape_errors_raise():
    with pytest.raises(ValueError):
        token_contrastive_loss(torch.randn(2, 4), torch.tensor([0]), torch.randn(2, 4), torch.tensor([0, 1]), 0.1)
    with pytest.raises(ValueError):
        token_contrastive_loss(torch.randn(1, 4), torch.tensor([0]), torch.randn(2, 5), torch.tensor([0, 1]), 0.1)
    with pytest.raises(ValueError):
        token_contrastive_loss(torch.randn(4), torch.tensor([0]), torch.randn(2, 4), torch.tensor([0, 1]), 0.1)
