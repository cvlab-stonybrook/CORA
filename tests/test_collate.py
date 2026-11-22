"""Tests of cora.data.collate and cora.data.samplers: Batch fields, alignment, aux batch, infinite loader."""
import copy
import pickle

import pytest
import torch
import torch.distributed as dist

from cora import constants as C
from cora.config import Config
from cora.data.batch import Batch, Conversation, Sample
from cora.data.collate import collate, make_collate
from cora.data.datasets import EvalDataset, LabeledDataset, UnlabeledDataset
from cora.data.samplers import InfiniteSampler, infinite_loader
from tests.test_conversation import get_tokenizer
from tests.test_datasets import CLIP, SAM, SEQ_LEN, mini, stage  # noqa: F401  (mini is a fixture)

ANSWER = f"It is {C.SEG_TOKEN}."


# ----------------------------------------------------------------------------------------------
# Synthetic samples (no dataset needed)
# ----------------------------------------------------------------------------------------------
def synthetic_sample(image_id, queries, kind="labeled", aux=None, refs=False, hw=(12, 20), answer=ANSWER):
    conversations = []
    for i, query in enumerate(queries):
        mask = torch.zeros(hw, dtype=torch.uint8)
        mask[i : i + 3, 2:8] = 1
        conversations.append(Conversation(
            query=query, answer=answer, mask=mask, class_id=i, set_name="conditional" if refs else "semantic",
            ref_pixels=torch.randn(3, CLIP, CLIP) if refs else None, ref_box=torch.rand(4) if refs else None,
            aux_queries=None if aux is None else list(aux), record_id=f"{image_id}:{i}",
        ))
    return Sample(image_id, torch.randn(3, SAM, SAM), torch.randn(3, CLIP, CLIP), (SAM // 2, SAM), hw, conversations, kind)


def synthetic_cfg(**reference):
    cfg = Config()
    for key, value in reference.items():
        setattr(cfg.model.reference, key, value)
    return cfg


def test_collate_fields_and_alignment_on_synthetic_samples():
    tok, spec = get_tokenizer(SEQ_LEN, CLIP)
    cfg = synthetic_cfg()
    a = synthetic_sample("a", ["Segment the sky.", "Can you segment the car in this image?"])
    b = synthetic_sample("b", ["What is the road in this image? Please output segmentation mask. " * 3], hw=(10, 30))
    batch = collate([a, b], tok, spec, cfg)
    assert isinstance(batch, Batch) and batch.kind == "labeled" and batch.num_images == 2 and batch.num_sequences == 3
    assert batch.image_ids == ["a", "b"] and batch.orig_hw == [(12, 20), (10, 30)] and batch.resize_hw == [(SAM // 2, SAM)] * 2
    assert batch.images_sam.shape == (2, 3, SAM, SAM) and batch.images_clip.shape == (2, 3, CLIP, CLIP)
    assert torch.equal(batch.seq_image, torch.tensor([0, 0, 1]))
    assert [m.shape for m in batch.masks] == [(12, 20), (12, 20), (10, 30)]
    assert torch.equal(batch.masks[1], a.conversations[1].mask)
    assert torch.equal(batch.class_ids, torch.tensor([0, 1, 0]))
    assert batch.set_names == ["semantic"] * 3 and batch.record_ids == ["a:0", "a:1", "b:0"]
    assert batch.queries[0] == "Segment the sky." and batch.answers == [ANSWER] * 3
    assert batch.is_long_query == [None] * 3 and batch.aux is None and batch.group is None and batch.confidence is None
    # sequences: right padded, mask and labels consistent
    n, length = batch.input_ids.shape
    assert n == 3 and batch.attention_mask.shape == batch.labels.shape == (n, length)
    real = batch.attention_mask.sum(1)
    assert real[2] > real[0] and length == int(real.max())
    for row in range(n):
        k = int(real[row])
        assert torch.all(batch.attention_mask[row, :k] == 1) and torch.all(batch.attention_mask[row, k:] == 0)
        assert torch.all(batch.input_ids[row, k:] == spec.pad_token_id) and torch.all(batch.labels[row, k:] == C.IGNORE_INDEX)
        assert batch.input_ids[row, 0] == spec.bos_token_id
        assert int((batch.input_ids[row] == spec.image_token_id).sum()) == SEQ_LEN
        assert int((batch.input_ids[row] == spec.seg_token_id).sum()) == 1
        supervised = batch.labels[row] != C.IGNORE_INDEX
        assert tok.convert_ids_to_tokens(batch.input_ids[row][supervised].tolist()) == ["▁It", "▁is", "▁", "<SSEG>", "▁.", "</s>"]
        assert batch.input_ids[row, k - 1] == spec.eos_token_id
    assert batch.ref_pixels.shape == (0, 3, CLIP, CLIP) and batch.ref_boxes.shape == (0, 4) and batch.ref_seq.shape == (0,)


def test_collate_references_in_placeholder_order():
    tok, spec = get_tokenizer(SEQ_LEN, CLIP)
    cond = "Segment the cars on the left of <REF>."
    a = synthetic_sample("a", [cond, cond.replace("cars", "buses")], refs=True)
    b = synthetic_sample("b", [cond], refs=True)
    for placement in ("inline", "prefix"):
        batch = collate([a, b], tok, spec, synthetic_cfg(placement=placement))
        assert batch.ref_pixels.shape == (3, 3, CLIP, CLIP) and batch.ref_boxes.shape == (3, 4)
        assert torch.equal(batch.ref_seq, torch.tensor([0, 1, 2]))
        assert torch.equal(batch.ref_pixels[1], a.conversations[1].ref_pixels) and torch.equal(batch.ref_boxes[2], b.conversations[0].ref_box)
        for row in batch.ref_seq.tolist():
            assert int((batch.input_ids[row] == spec.ref_mask_token_id).sum()) == int((batch.input_ids[row] == spec.ref_box_token_id).sum()) == 1
        assert batch.queries[0] == cond  # the stored query (with <REF>) is kept for logging
    text = collate([a, b], tok, spec, synthetic_cfg(mode="text"))
    assert text.ref_pixels.shape[0] == 0 and not (text.input_ids == spec.ref_mask_token_id).any()


def test_collate_unlabeled_builds_aux_batch():
    tok, spec = get_tokenizer(SEQ_LEN, CLIP)
    aux = ["Segment the first rephrasing.", "Segment the second rephrasing.", "Segment the third rephrasing."]
    a = synthetic_sample("a", ["Segment the main query.", "Segment another main query."], kind="unlabeled", aux=aux)
    for c in a.conversations:
        c.confidence = torch.rand(12, 20)
    b = synthetic_sample("b", ["Segment the b query."], kind="unlabeled", aux=aux, hw=(12, 20))
    b.conversations[0].confidence = None
    batch = collate([a, b], tok, spec, synthetic_cfg())
    assert batch.kind == "unlabeled" and batch.num_sequences == 3
    assert batch.confidence[0] is a.conversations[0].confidence and batch.confidence[2] is None
    aux_batch = batch.aux
    assert aux_batch.kind == "aux" and aux_batch.num_sequences == 3 * 3 and aux_batch.aux is None
    assert torch.equal(aux_batch.group, torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]))
    assert aux_batch.images_sam is batch.images_sam and aux_batch.images_clip is batch.images_clip
    assert aux_batch.image_ids == batch.image_ids and aux_batch.orig_hw == batch.orig_hw
    assert torch.equal(aux_batch.seq_image, batch.seq_image[aux_batch.group])
    assert torch.equal(aux_batch.class_ids, batch.class_ids[aux_batch.group])
    assert aux_batch.queries == aux * 3
    for j in range(9):
        parent = int(aux_batch.group[j])
        assert torch.equal(aux_batch.masks[j], batch.masks[parent])
        assert aux_batch.answers[j] == batch.answers[parent]
        assert int((aux_batch.input_ids[j] == spec.seg_token_id).sum()) == 1
        sup = aux_batch.labels[j] != C.IGNORE_INDEX
        assert tok.convert_ids_to_tokens(aux_batch.input_ids[j][sup].tolist()) == ["▁It", "▁is", "▁", "<SSEG>", "▁.", "</s>"]
    assert aux_batch.ref_pixels.shape[0] == 0
    # the aux sequences differ from each other only by their query text
    assert len({tuple(aux_batch.input_ids[j].tolist()) for j in range(3)}) == 3


def test_collate_errors():
    tok, spec = get_tokenizer(SEQ_LEN, CLIP)
    cfg = synthetic_cfg()
    with pytest.raises(ValueError, match="empty"):
        collate([], tok, spec, cfg)
    with pytest.raises(ValueError, match="one kind"):
        collate([synthetic_sample("a", ["Segment the sky."]), synthetic_sample("b", ["Segment the sky."], kind="eval")], tok, spec, cfg)
    with pytest.raises(ValueError, match="aux_queries"):
        collate([synthetic_sample("a", ["Segment the sky."], kind="unlabeled")], tok, spec, cfg)
    no_ref = synthetic_sample("a", ["Segment the cars on the left of <REF>."])
    with pytest.raises(ValueError, match="ref_pixels"):
        collate([no_ref], tok, spec, cfg)
    with pytest.raises(ValueError, match="no conversation"):
        empty = synthetic_sample("a", ["x"])
        empty.conversations = []
        collate([empty], tok, spec, cfg)
    with pytest.raises(ValueError, match="exactly one <SSEG>"):
        collate([synthetic_sample("a", ["Segment the sky."], answer="It is here.")], tok, spec, cfg)


def test_make_collate_is_picklable():
    tok, spec = get_tokenizer(SEQ_LEN, CLIP)
    fn = pickle.loads(pickle.dumps(make_collate(tok, spec, synthetic_cfg())))
    batch = fn([synthetic_sample("a", ["Segment the sky."])])
    assert isinstance(batch, Batch) and batch.num_sequences == 1


# ----------------------------------------------------------------------------------------------
# Real mini data
# ----------------------------------------------------------------------------------------------
@pytest.mark.data
def test_labeled_batch_from_real_samples(mini):  # noqa: F811
    cfg = copy.deepcopy(mini.cfg)
    ds = LabeledDataset(cfg, stage([C.SET_CONDITIONAL]), mini.spec, seed=1)
    samples = [ds[0], ds[1]]
    batch = collate(samples, mini.tokenizer, mini.spec, cfg)
    n = sum(len(s.conversations) for s in samples)
    assert batch.num_sequences == n and batch.num_images == 2 and batch.ref_pixels.shape == (n, 3, CLIP, CLIP)
    assert torch.equal(batch.ref_seq, torch.arange(n))
    flat = [c for s in samples for c in s.conversations]
    assert batch.record_ids == [c.record_id for c in flat]
    assert all(torch.equal(m, c.mask) for m, c in zip(batch.masks, flat))
    assert torch.equal(batch.class_ids, torch.tensor([c.class_id for c in flat]))
    assert torch.equal(batch.seq_image, torch.tensor([0] * len(samples[0].conversations) + [1] * len(samples[1].conversations)))
    batch.to("cpu")  # the Batch container accepts what collate produced


@pytest.mark.data
def test_unlabeled_batch_from_real_samples(mini):  # noqa: F811
    ds = UnlabeledDataset(mini.cfg, mini.spec, seed=0, allow_partial=True)
    batch = collate([ds[0], ds[1]], mini.tokenizer, mini.spec, mini.cfg)
    k = mini.cfg.unlabeled.num_queries
    assert batch.kind == "unlabeled" and batch.num_sequences == 2 and batch.aux.num_sequences == 2 * (k - 1)
    assert torch.equal(batch.aux.group, torch.arange(2).repeat_interleave(k - 1))
    tok = mini.tokenizer
    for parent in range(2):
        rows = (batch.aux.group == parent).nonzero().flatten().tolist()
        queries = [tok.decode(batch.aux.input_ids[r], skip_special_tokens=True).split("USER:")[1].split("ASSISTANT:")[0].strip() for r in rows]
        trained = tok.decode(batch.input_ids[parent], skip_special_tokens=True).split("USER:")[1].split("ASSISTANT:")[0].strip()
        assert len(set(queries + [trained])) == k  # K - 1 rephrasings, all distinct and different from the trained query
        assert len({int(batch.aux.class_ids[r]) for r in rows}) == 1 and int(batch.aux.class_ids[rows[0]]) == int(batch.class_ids[parent])
    # the confidence map is the weight of the confidence-threshold baseline only: not loaded by default
    assert batch.confidence is None or all(c is None for c in batch.confidence)
    conf_cfg = copy.deepcopy(mini.cfg)
    conf_cfg.unlabeled.weighting = "confidence"
    ds = UnlabeledDataset(conf_cfg, mini.spec, seed=0, allow_partial=True)
    batch = collate([ds[0], ds[1]], mini.tokenizer, mini.spec, conf_cfg)
    assert all(c.shape == (1024, 2048) for c in batch.confidence)


@pytest.mark.data
def test_eval_batch_from_real_samples(mini):  # noqa: F811
    ds = EvalDataset(mini.cfg, mini.spec, [C.SET_SEMANTIC], "val")
    batch = collate([ds[0]], mini.tokenizer, mini.spec, mini.cfg)
    assert batch.kind == "eval" and batch.answers == [ANSWER] and batch.aux is None


# ----------------------------------------------------------------------------------------------
# Samplers and the infinite loader
# ----------------------------------------------------------------------------------------------
def take(sampler, count):
    out = []
    for index in sampler:
        out.append(index)
        if len(out) == count:
            return out


def test_infinite_sampler_covers_every_item_each_pass_and_never_repeats_an_index():
    seq = take(InfiniteSampler(7, seed=1), 21)
    for epoch in range(3):
        assert sorted(i - epoch * 7 for i in seq[epoch * 7 : (epoch + 1) * 7]) == list(range(7))
    assert len(set(seq)) == 21
    assert take(InfiniteSampler(7, seed=1), 21) == seq and take(InfiniteSampler(7, seed=2), 21) != seq
    assert seq[:7] != sorted(seq[:7]) or seq[7:14] != sorted(seq[7:14])  # shuffled


def test_infinite_sampler_ranks_are_disjoint_and_equal_sized():
    ranks = [take(InfiniteSampler(10, 3, rank=r, world_size=3), 8) for r in range(3)]
    flat = [i for r in ranks for i in r]
    assert len(set(flat)) == len(flat)  # no index is used by two ranks
    assert {i % 10 for r in ranks for i in r[:4]} == set(range(10))  # the first pass covers every item
    assert all(len(r) == 8 for r in ranks)
    with pytest.raises(ValueError):
        InfiniteSampler(0, 0)
    with pytest.raises(ValueError):
        InfiniteSampler(5, 0, rank=2, world_size=2)


class CountingDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return index


def test_infinite_loader_runs_forever_and_reshuffles():
    loader = infinite_loader(CountingDataset(), 2, lambda items: items, 0, seed=0, distributed=False)
    items = [i for _ in range(9) for i in next(loader)]  # three passes
    assert len(items) == 18 and len(set(items)) == 18
    assert sorted(i % 6 for i in items[:6]) == list(range(6))
    assert [i % 6 for i in items[:6]] != [i % 6 for i in items[6:12]]


def test_infinite_loader_distributed_needs_a_process_group(tmp_path):
    with pytest.raises(RuntimeError, match="process group"):
        next(infinite_loader(CountingDataset(), 2, lambda items: items, 0, 0, True))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        assert len(next(infinite_loader(CountingDataset(), 2, lambda items: items, 0, 0, True))) == 2
    finally:
        dist.destroy_process_group()


def test_infinite_loader_argument_errors():
    with pytest.raises(ValueError, match="batch_size"):
        next(infinite_loader(CountingDataset(), 0, lambda x: x, 0, 0, False))
    with pytest.raises(ValueError, match="num_workers"):
        next(infinite_loader(CountingDataset(), 1, lambda x: x, -1, 0, False))


@pytest.mark.data
def test_infinite_loader_with_two_workers_on_real_data(mini):  # noqa: F811
    st = stage([C.SET_SEMANTIC, C.SET_ATTRIBUTE, C.SET_CONDITIONAL])

    def run(num_workers):
        ds = LabeledDataset(mini.cfg, st, mini.spec, seed=7)
        loader = infinite_loader(ds, 2, make_collate(mini.tokenizer, mini.spec, mini.cfg), num_workers, seed=7, distributed=False)
        return [next(loader) for _ in range(8)]  # more batches than one pass over the 100 images? no: a stream, not an epoch

    first, second = run(2), run(2)
    assert all(isinstance(b, Batch) and b.num_images == 2 and b.kind == "labeled" for b in first)
    assert [b.record_ids for b in first] == [b.record_ids for b in second]  # reproducible with two workers
    ids = [r for b in first for r in b.record_ids]
    assert len(set(ids)) > 10  # draws differ from batch to batch (every index has its own seed)
    unl = UnlabeledDataset(mini.cfg, mini.spec, seed=7, allow_partial=True)
    loader = infinite_loader(unl, 2, make_collate(mini.tokenizer, mini.spec, mini.cfg), 2, seed=7, distributed=False)
    batches = [next(loader) for _ in range(3)]  # 3 batches of 2 images: more than one pass over the 3 pseudo-labelled images
    assert all(b.kind == "unlabeled" and b.aux is not None and b.aux.num_sequences == 6 * b.num_sequences for b in batches)
