"""Tests of cora.losses.token_bank."""
import pytest
import torch

from cora.losses.token_bank import TokenBank


def _feat(values, dim=2):
    return torch.tensor(values, dtype=torch.float32).unsqueeze(1).repeat(1, dim)


def test_empty_bank():
    bank = TokenBank(3, 4, 5)
    feats, classes = bank.get()
    assert feats.shape == (0, 4) and classes.shape == (0,) and classes.dtype == torch.long
    assert all(bank.count(c) == 0 for c in range(3))


def test_enqueue_and_get_return_only_filled_slots_in_fifo_order():
    bank = TokenBank(3, 2, 4)
    bank.enqueue(_feat([1, 2, 3]), torch.tensor([0, 1, 0]))
    feats, classes = bank.get()
    assert classes.tolist() == [0, 0, 1]
    assert feats[:, 0].tolist() == [1.0, 3.0, 2.0]
    assert bank.count(0) == 2 and bank.count(1) == 1 and bank.count(2) == 0


def test_newest_overwrite_oldest():
    bank = TokenBank(2, 2, 3)
    bank.enqueue(_feat([1, 2, 3]), torch.zeros(3, dtype=torch.long))
    bank.enqueue(_feat([4, 5]), torch.zeros(2, dtype=torch.long))
    feats, classes = bank.get()
    assert feats[:, 0].tolist() == [3.0, 4.0, 5.0]  # oldest first
    assert bank.count(0) == 3 and classes.tolist() == [0, 0, 0]
    bank.enqueue(_feat([6]), torch.zeros(1, dtype=torch.long))
    assert bank.get()[0][:, 0].tolist() == [4.0, 5.0, 6.0]


def test_more_new_entries_than_capacity_keeps_newest():
    bank = TokenBank(2, 2, 3)
    bank.enqueue(_feat([0]), torch.tensor([1]))
    bank.enqueue(_feat([1, 2, 3, 4, 5]), torch.ones(5, dtype=torch.long))
    feats, _ = bank.get()
    assert feats[:, 0].tolist() == [3.0, 4.0, 5.0]
    assert bank.count(1) == 3
    bank.enqueue(_feat([6]), torch.tensor([1]))
    assert bank.get()[0][:, 0].tolist() == [4.0, 5.0, 6.0]


def test_negative_class_ids_are_ignored():
    bank = TokenBank(2, 2, 3)
    bank.enqueue(_feat([1, 2, 3]), torch.tensor([-1, 1, -1]))
    feats, classes = bank.get()
    assert feats[:, 0].tolist() == [2.0] and classes.tolist() == [1]
    bank.enqueue(_feat([9]), torch.tensor([-1]))
    assert bank.get()[0].shape[0] == 1


def test_enqueue_detaches_and_stores_float32():
    bank = TokenBank(2, 3, 4)
    x = torch.randn(2, 3, requires_grad=True)
    bank.enqueue(x, torch.tensor([0, 1]))
    assert not bank.features.requires_grad
    bank.enqueue(x.detach().bfloat16(), torch.tensor([0, 1]))
    assert bank.features.dtype == torch.float32 and bank.count(0) == 2


def test_state_dict_round_trip_continues_the_queue():
    bank = TokenBank(2, 2, 3)
    bank.enqueue(_feat([1, 2, 3, 4]), torch.tensor([0, 0, 0, 1]))
    state = bank.state_dict()
    assert {"features", "filled", "next_slot"} <= set(state)
    restored = TokenBank(2, 2, 3)
    restored.load_state_dict(state)
    for a, b in zip(bank.get(), restored.get()):
        assert torch.equal(a, b)
    bank.enqueue(_feat([7]), torch.tensor([0]))
    restored.enqueue(_feat([7]), torch.tensor([0]))
    assert torch.equal(bank.get()[0], restored.get()[0])
    assert restored.get()[0][restored.get()[1] == 0][:, 0].tolist() == [2.0, 3.0, 7.0]


def test_bank_follows_module_dtype_and_device_moves():
    bank = TokenBank(2, 2, 3).to(torch.float64)
    bank.enqueue(_feat([1]), torch.tensor([0]))
    assert bank.get()[0].dtype == torch.float64


@pytest.mark.parametrize("args", [(0, 2, 3), (2, 0, 3), (2, 2, 0)])
def test_invalid_construction(args):
    with pytest.raises(ValueError):
        TokenBank(*args)


def test_invalid_enqueue_and_count():
    bank = TokenBank(2, 2, 3)
    with pytest.raises(ValueError):
        bank.enqueue(torch.zeros(1, 3), torch.tensor([0]))
    with pytest.raises(ValueError):
        bank.enqueue(torch.zeros(2, 2), torch.tensor([0]))
    with pytest.raises(ValueError):
        bank.enqueue(torch.zeros(1, 2), torch.tensor([2]))
    with pytest.raises(ValueError):
        bank.count(2)
    bank.enqueue(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long))
    assert bank.get()[0].shape[0] == 0
