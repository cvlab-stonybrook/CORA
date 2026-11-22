"""Tests of cora.data.datasets on a miniature data folder built from real Cityscapes images (marker ``data``).

The folder (objects, instruction sets, debug pseudo-labels, shipped query database) is built in a temporary
directory by the real generators of this repository, so the datasets are tested against the real file formats.
Token geometry is tiny (4 image tokens, 28 px CLIP, 64 px SAM frame) to keep the tests fast.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cora import constants as C
from cora.config import Config, StageConfig
from cora.data.adapters import get_adapter
from cora.data.datasets import (
    EVAL_ANSWER, EvalDataset, LabeledDataset, UnlabeledDataset, eval_group, load_instruction_records, record_style, stream_rng,
)
from cora.data.objects import ensure_objects
from cora.data.query_db import QueryDB
from cora.data.schema import write_jsonl
from cora.data.templates import ANSWER_LIST
from cora.data.transforms import anchor_box, anchor_crop, clip_preprocess
from cora.instructions.offline import build_offline_records
from cora.instructions.semantic import build_semantic_records
from cora.instructions.templated_conditional import build_templated_conditional
from cora.pseudo.io import write_prediction
from tests.test_conversation import get_tokenizer

pytestmark = pytest.mark.data

N_LABELED, N_UNLABELED, N_VAL = 4, 3, 2
SEQ_LEN, CLIP, SAM = 4, 28, 64


def mini_config(data_dir) -> Config:
    cfg = Config()
    cfg.paths.data_dir = str(data_dir)
    cfg.data.partition = "1_30"
    cfg.data.ssl_method = "debug"
    cfg.data.sam_image_size = SAM
    cfg.data.queries_per_image = 3
    return cfg


@pytest.fixture(scope="module")
def mini(tmp_path_factory, cityscapes_root):
    """Miniature data folder: instruction sets of 4 labeled and 2 val images, pseudo-labels of 3 unlabeled images."""
    root = tmp_path_factory.mktemp("mini_data")
    cfg = mini_config(root)
    cfg.paths.cityscapes_root = str(cityscapes_root)
    adapter = get_adapter(cfg)
    labeled = adapter.partition("1_30", "labeled")[:N_LABELED]
    unlabeled = adapter.partition("1_30", "unlabeled")[:N_UNLABELED]
    val = adapter.val_list()[:N_VAL]
    for split, ids in (("train", labeled), ("val", val)):
        write_jsonl(cfg.instructions_path(split, "semantic"), build_semantic_records(cfg, adapter, split, ids))
        write_jsonl(cfg.instructions_path(split, "attribute"), build_offline_records(cfg, adapter, split, ids, [C.SET_ATTRIBUTE]))
        write_jsonl(cfg.instructions_path(split, "conditional"), build_offline_records(cfg, adapter, split, ids, [C.SET_CONDITIONAL]))
        write_jsonl(cfg.instructions_path(split, "conditional", "templated"), build_templated_conditional(cfg, adapter, split, ids))
    rng = np.random.default_rng(0)
    for image_id in unlabeled:
        label = adapter.load_label(image_id)
        write_prediction(cfg.pseudo_dir, image_id, label, rng.uniform(0.3, 1.0, label.shape).astype(np.float32))
    tokenizer, spec = get_tokenizer(SEQ_LEN, CLIP)
    return SimpleNamespace(cfg=cfg, adapter=adapter, labeled=labeled, unlabeled=unlabeled, val=val, tokenizer=tokenizer, spec=spec)


def stage(sets, **kw) -> StageConfig:
    return StageConfig(name="t", labeled_sets=list(sets), use_reference=C.SET_CONDITIONAL in sets, **kw)


def records_by_id(cfg, split="train"):
    out = {}
    for set_name in C.LABELED_SETS:
        for record in load_instruction_records(cfg, set_name, split):
            out[record.id] = record
    return out


# ----------------------------------------------------------------------------------------------
# Instruction files
# ----------------------------------------------------------------------------------------------
def test_instruction_files_include_variants(mini):
    files = [f.name for f in mini.cfg.instruction_files("train", "conditional")]
    assert files == ["conditional.jsonl", "conditional_templated.jsonl"]
    records = load_instruction_records(mini.cfg, C.SET_CONDITIONAL, "train")
    assert {r.meta["generator"] for r in records} == {"offline", "rule"}


# ----------------------------------------------------------------------------------------------
# LabeledDataset
# ----------------------------------------------------------------------------------------------
def check_sample_geometry(sample, mini):
    assert sample.image_sam.shape == (3, SAM, SAM) and sample.image_sam.dtype == torch.float32
    assert sample.image_clip.shape == (3, CLIP, CLIP)
    assert sample.orig_hw == (1024, 2048) and sample.resize_hw == (SAM // 2, SAM)
    assert torch.all(sample.image_sam[:, SAM // 2 :] == 0)  # SAM padding at the bottom


def test_labeled_stage1_samples_align_masks_classes_and_answers(mini):
    ds = LabeledDataset(mini.cfg, stage([C.SET_SEMANTIC, C.SET_ATTRIBUTE]), mini.spec, seed=3)
    assert len(ds) == 100  # nominal epoch = labeled images of the partition
    by_id = records_by_id(mini.cfg)
    seen_sets = set()
    for index in range(30):
        sample = ds[index]
        check_sample_geometry(sample, mini)
        assert sample.kind == "labeled" and sample.image_id in mini.labeled
        assert 1 <= len(sample.conversations) <= 3
        label = mini.adapter.load_label(sample.image_id)
        objects = ensure_objects(mini.cfg, mini.adapter, "train", sample.image_id)
        assert len({c.record_id for c in sample.conversations}) == len(sample.conversations)
        assert len({c.set_name for c in sample.conversations}) == 1  # one set per draw
        for conv in sample.conversations:
            record = by_id[conv.record_id]
            seen_sets.add(conv.set_name)
            assert record.image_id == sample.image_id and conv.set_name == record.set and conv.class_id == record.target.label_id
            assert conv.answer in ANSWER_LIST
            assert conv.answer.count(C.SEG_TOKEN) == 1 and C.REF_TOKEN not in conv.query
            assert conv.mask.dtype == torch.uint8 and conv.mask.shape == (1024, 2048)
            assert set(conv.mask.unique().tolist()) <= {0, 1} and conv.mask.sum() > 0
            if record.target.region_ids:
                expected = objects.mask(record.target.region_ids)
            else:
                expected = label == record.target.label_id
            assert np.array_equal(conv.mask.numpy().astype(bool), expected)
            assert conv.ref_pixels is None and conv.ref_box is None
            if record.set == C.SET_SEMANTIC:
                assert conv.query.lower().count(record.target.label_name) >= 1
        if sample.conversations[0].set_name == C.SET_SEMANTIC:
            assert len({c.class_id for c in sample.conversations}) == len(sample.conversations)  # distinct classes
    assert seen_sets == {C.SET_SEMANTIC, C.SET_ATTRIBUTE}


def test_semantic_queries_get_fresh_templates(mini):
    ds = LabeledDataset(mini.cfg, stage([C.SET_SEMANTIC]), mini.spec, seed=0)
    stored = {r.id: r.query for r in load_instruction_records(mini.cfg, C.SET_SEMANTIC, "train")}
    queries = {}
    for index in range(60):
        for conv in ds[index].conversations:
            queries.setdefault(conv.record_id, set()).add(conv.query)
    assert any(len(q) > 1 for q in queries.values())  # the same record is asked with different templates
    assert any(conv_query != stored[rid] for rid, qs in queries.items() for conv_query in qs)


def test_labeled_stage2_conditional_has_reference_inputs(mini):
    ds = LabeledDataset(mini.cfg, stage([C.SET_CONDITIONAL]), mini.spec, seed=1)
    by_id = records_by_id(mini.cfg)
    generators = set()
    for index in range(30):
        sample = ds[index]
        label_objects = ensure_objects(mini.cfg, mini.adapter, "train", sample.image_id)
        image = mini.adapter.load_image(sample.image_id)
        for conv in sample.conversations:
            record = by_id[conv.record_id]
            generators.add(record.meta["generator"])
            assert conv.query.count(C.REF_TOKEN) == 1 and conv.set_name == C.SET_CONDITIONAL
            anchor = label_objects.mask(record.anchor.region_ids)
            assert conv.ref_pixels.shape == (3, CLIP, CLIP) and conv.ref_box.shape == (4,)
            assert torch.allclose(conv.ref_pixels, clip_preprocess(anchor_crop(image, anchor), CLIP, "pad"))
            assert torch.allclose(conv.ref_box, anchor_box(anchor, SAM))
            assert not (conv.mask.numpy().astype(bool) & anchor).any()  # the anchor is not part of the target
    assert generators == {"offline", "rule"}  # both files of the set are used


def test_text_reference_mode_makes_no_reference_inputs(mini):
    cfg = copy.deepcopy(mini.cfg)
    cfg.model.reference.mode = "text"
    ds = LabeledDataset(cfg, stage([C.SET_CONDITIONAL]), mini.spec)
    sample = ds[0]
    assert all(c.ref_pixels is None and c.ref_box is None and C.REF_TOKEN in c.query for c in sample.conversations)


def test_set_weights_select_sets(mini):
    ds = LabeledDataset(mini.cfg, stage([C.SET_SEMANTIC, C.SET_ATTRIBUTE], set_weights=[0.0, 1.0]), mini.spec)
    assert {ds[i].conversations[0].set_name for i in range(10)} == {C.SET_ATTRIBUTE}
    with pytest.raises(ValueError, match="set_weights"):
        LabeledDataset(mini.cfg, stage([C.SET_SEMANTIC], set_weights=[0.0]), mini.spec)


def test_labeled_sampling_is_reproducible_under_a_fixed_seed(mini):
    st = stage([C.SET_SEMANTIC, C.SET_ATTRIBUTE, C.SET_CONDITIONAL])
    a, b, other = (LabeledDataset(mini.cfg, st, mini.spec, seed=s) for s in (5, 5, 6))

    def signature(ds, i):
        s = ds[i]
        return s.image_id, [(c.record_id, c.query, c.answer) for c in s.conversations]

    same = [signature(a, i) == signature(b, i) for i in range(12)]
    differ = [signature(a, i) != signature(other, i) for i in range(12)]
    assert all(same) and sum(differ) >= 8
    assert signature(a, 3) == signature(a, 3)
    assert torch.equal(a[2].conversations[0].mask, b[2].conversations[0].mask)
    assert len({signature(a, i)[0] for i in range(12)}) > 1  # different indices draw different images
    assert stream_rng(1, 2).random() == stream_rng(1, 2).random() != stream_rng(1, 3).random()
    # the draw does not depend on the loader worker that serves the index (a resumed stage uses other workers)
    from torch.utils import data as torch_data

    reference = stream_rng(1, 2).random()
    worker = type("Info", (), {"id": 3, "seed": 99, "num_workers": 8})()
    original = torch_data.get_worker_info
    torch_data.get_worker_info = lambda: worker
    try:
        assert stream_rng(1, 2).random() == reference
    finally:
        torch_data.get_worker_info = original


def test_labeled_index_beyond_length_is_a_seed(mini):
    ds = LabeledDataset(mini.cfg, stage([C.SET_SEMANTIC]), mini.spec)
    assert ds[10_000].image_id in mini.labeled
    with pytest.raises(IndexError):
        ds[-1]


def test_missing_instruction_file_is_an_error_naming_the_file(mini, tmp_path):
    cfg = copy.deepcopy(mini.cfg)
    cfg.paths.data_dir = str(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match=r"attribute\.jsonl"):
        LabeledDataset(cfg, stage([C.SET_ATTRIBUTE]), mini.spec)
    cfg2 = copy.deepcopy(mini.cfg)
    cfg2.paths.data_dir = str(tmp_path / "partial")
    write_jsonl(cfg2.instructions_path("train", "semantic"), build_semantic_records(cfg2, mini.adapter, "train", mini.labeled[:1]))
    with pytest.raises(FileNotFoundError, match=r"conditional\.jsonl"):
        LabeledDataset(cfg2, stage([C.SET_SEMANTIC, C.SET_CONDITIONAL]), mini.spec)


def test_llm_written_records_without_meta_model_load(tmp_path):
    """The provided attribute / conditional files carry no ``meta["model"]``; reading them must not need it."""
    cfg = mini_config(tmp_path)
    attribute = {
        "id": "train/a/a_1:attribute:13_1:1", "dataset": "cityscapes", "split": "train", "image_id": "train/a/a_1",
        "set": "attribute", "query": "Can you segment the compact four-wheeled vehicle parked at the curb?",
        "target": {"label_id": 13, "label_name": "car", "region_ids": ["13_1"]}, "anchor": None, "relation": None,
        "description": "A compact four-wheeled vehicle parked at the curb.",
        "meta": {"generator": "llm", "style": "attribute", "question_number": 1, "query_source": "rewrite", "seed": 0},
    }
    conditional = {
        "id": "train/a/a_1:conditional:11_1:13_1:1", "dataset": "cityscapes", "split": "train", "image_id": "train/a/a_1",
        "set": "conditional", "query": f"Segment the car that is parked on the left side of {C.REF_TOKEN}.",
        "target": {"label_id": 13, "label_name": "car", "region_ids": ["13_1"]},
        "anchor": {"label_id": 11, "label_name": "person", "region_ids": ["11_1"]}, "relation": "to the left of",
        "description": None, "meta": {"generator": "llm", "style": "llm_descriptive", "seed": 0},
    }
    for record in (attribute, conditional):
        assert "model" not in record["meta"]
        path = cfg.instructions_path("train", record["set"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    for set_name, expected in ((C.SET_ATTRIBUTE, attribute), (C.SET_CONDITIONAL, conditional)):
        (record,) = load_instruction_records(cfg, set_name, "train")
        record.validate()
        assert record.id == expected["id"] and "model" not in record.meta and record.meta["generator"] == "llm"
        assert record_style(record) == "llm" and eval_group(record) == set_name


def test_records_outside_the_partition_are_an_error(mini, tmp_path):
    cfg = copy.deepcopy(mini.cfg)
    cfg.paths.data_dir = str(tmp_path / "other")
    outside = mini.adapter.partition("1_30", "unlabeled")[:1]
    write_jsonl(cfg.instructions_path("train", "semantic"), build_semantic_records(cfg, mini.adapter, "train", outside))
    with pytest.raises(ValueError, match="no record for the labeled images"):
        LabeledDataset(cfg, stage([C.SET_SEMANTIC]), mini.spec)


# ----------------------------------------------------------------------------------------------
# UnlabeledDataset
# ----------------------------------------------------------------------------------------------
def test_unlabeled_sample_structure(mini):
    ds = UnlabeledDataset(mini.cfg, mini.spec, seed=2, allow_partial=True)
    assert len(ds) == N_UNLABELED
    db = QueryDB.for_config(mini.cfg)
    ucfg = mini.cfg.unlabeled
    for index in range(9):
        sample = ds[index]
        check_sample_geometry(sample, mini)
        assert sample.kind == "unlabeled" and sample.image_id == mini.unlabeled[index % N_UNLABELED]
        label = mini.adapter.load_label(sample.image_id)  # the debug pseudo-label is the ground truth
        assert len(sample.conversations) == ucfg.classes_per_image == 1
        conv = sample.conversations[0]
        assert conv.set_name == C.SET_UNLABELED and conv.record_id.startswith(sample.image_id)
        assert (label == conv.class_id).sum() >= ucfg.min_class_pixels
        assert np.array_equal(conv.mask.numpy(), (label == conv.class_id).astype(np.uint8))
        queries = [conv.query] + conv.aux_queries
        assert len(conv.aux_queries) == ucfg.num_queries - 1 == 6
        assert len(set(queries)) == ucfg.num_queries  # K distinct queries
        assert set(queries) <= set(db.queries(conv.class_id))
        assert conv.answer.count(C.SEG_TOKEN) == 1
        assert not any(C.REF_TOKEN in q for q in queries) and conv.ref_pixels is None
        assert conv.confidence is None  # consistency weighting (default) does not use the confidence map


def test_unlabeled_confidence_is_attached_only_for_confidence_weighting(mini, monkeypatch):
    cfg = copy.deepcopy(mini.cfg)
    cfg.unlabeled.weighting = "confidence"
    conv = UnlabeledDataset(cfg, mini.spec, allow_partial=True)[0].conversations[0]
    assert conv.confidence.shape == (1024, 2048) and conv.confidence.dtype == torch.float32
    assert 0.0 <= conv.confidence.min() and conv.confidence.max() <= 1.0
    # under the default weighting the confidence PNG is never decoded
    import cora.data.datasets as datasets_module

    def forbidden(*args, **kwargs):
        raise AssertionError("the confidence map must not be read")

    monkeypatch.setattr(datasets_module, "read_prediction", forbidden)
    ds = UnlabeledDataset(mini.cfg, mini.spec, allow_partial=True)
    assert all(ds[i].conversations[0].confidence is None for i in range(3))


def test_unlabeled_confidence_weighting_without_any_confidence_file_is_an_error(mini, tmp_path):
    cfg = copy.deepcopy(mini.cfg)
    cfg.paths.data_dir = str(tmp_path / "noconf")
    cfg.unlabeled.weighting = "confidence"
    write_prediction(cfg.pseudo_dir, mini.unlabeled[0], mini.adapter.load_label(mini.unlabeled[0]), None)
    with pytest.raises(FileNotFoundError, match="confidence maps"):
        UnlabeledDataset(cfg, mini.spec, allow_partial=True)
    cfg.unlabeled.weighting = "consistency"
    UnlabeledDataset(cfg, mini.spec, allow_partial=True)


def test_unlabeled_rejects_a_query_database_with_a_wrong_class_entry(mini, tmp_path):
    db = QueryDB.for_config(mini.cfg)
    data = db.to_dict()
    data["classes"]["0"]["queries"][0] = "Segment the car"  # class 0 is road
    cfg = copy.deepcopy(mini.cfg)
    cfg.paths.data_dir = str(tmp_path / "baddb")
    cfg.query_db_path.parent.mkdir(parents=True)
    cfg.query_db_path.write_text(json.dumps(data), encoding="utf-8")
    cfg.pseudo_dir.mkdir(parents=True)
    write_prediction(cfg.pseudo_dir, mini.unlabeled[0], mini.adapter.load_label(mini.unlabeled[0]), None)
    with pytest.raises(ValueError, match="invalid query database"):
        UnlabeledDataset(cfg, mini.spec, allow_partial=True)


def test_unlabeled_several_classes_per_image_are_distinct(mini):
    cfg = copy.deepcopy(mini.cfg)
    cfg.unlabeled.classes_per_image = 3
    sample = UnlabeledDataset(cfg, mini.spec, allow_partial=True)[0]
    classes = [c.class_id for c in sample.conversations]
    assert len(classes) == 3 and len(set(classes)) == 3
    assert all(c.confidence is None for c in sample.conversations)


def test_unlabeled_sampling_is_reproducible(mini):
    a, b, other = (UnlabeledDataset(mini.cfg, mini.spec, seed=s, allow_partial=True) for s in (4, 4, 9))
    sig = lambda ds, i: [(c.class_id, c.query, tuple(c.aux_queries), c.answer) for c in ds[i].conversations]  # noqa: E731
    assert [sig(a, i) for i in range(6)] == [sig(b, i) for i in range(6)]
    assert [sig(a, i) for i in range(6)] != [sig(other, i) for i in range(6)]


def test_unlabeled_resamples_when_no_class_is_large_enough(mini, tmp_path):
    cfg = copy.deepcopy(mini.cfg)
    cfg.unlabeled.min_class_pixels = 10**9
    with pytest.raises(ValueError, match="no image tried"):
        UnlabeledDataset(cfg, mini.spec, allow_partial=True)[0]
    # an image whose pseudo-label has no sufficiently large class makes the dataset take another image
    cfg2 = copy.deepcopy(mini.cfg)
    cfg2.paths.data_dir = str(tmp_path / "pseudo")
    write_prediction(cfg2.pseudo_dir, mini.unlabeled[0], np.full((1024, 2048), 255, np.uint8), None)
    write_prediction(cfg2.pseudo_dir, mini.unlabeled[1], mini.adapter.load_label(mini.unlabeled[1]), None)
    ds = UnlabeledDataset(cfg2, mini.spec, allow_partial=True)
    assert len(ds) == 2
    sample = ds[0]  # index 0 is the all-void image
    assert sample.image_id == mini.unlabeled[1] and sample.conversations[0].confidence is None


def test_unlabeled_partial_pseudo_labels_are_an_error_unless_allowed(mini, caplog):
    """3 pseudo-labels for the 2875 unlabeled images of the partition must not silently shrink the dataset."""
    with pytest.raises(FileNotFoundError, match=r"have no pseudo-label.*allow_partial"):
        UnlabeledDataset(mini.cfg, mini.spec)
    with caplog.at_level("WARNING", logger="cora.data.datasets"):
        ds = UnlabeledDataset(mini.cfg, mini.spec, allow_partial=True)
    assert len(ds) == len(mini.unlabeled) and "allow_partial" in caplog.text


def test_unlabeled_warns_when_only_some_confidence_files_exist(mini, tmp_path, caplog):
    cfg = copy.deepcopy(mini.cfg)
    cfg.unlabeled.weighting = "confidence"
    cfg.paths.data_dir = str(tmp_path / "mixed")
    label = mini.adapter.load_label(mini.unlabeled[0])
    conf = np.full(label.shape, 0.5, np.float32)
    write_prediction(cfg.pseudo_dir, mini.unlabeled[0], label, conf)
    write_prediction(cfg.pseudo_dir, mini.unlabeled[1], label, None)
    with caplog.at_level("WARNING", logger="cora.data.datasets"):
        UnlabeledDataset(cfg, mini.spec, allow_partial=True)
    assert "no confidence file" in caplog.text


def test_unlabeled_errors(mini, tmp_path):
    cfg = copy.deepcopy(mini.cfg)
    cfg.paths.data_dir = str(tmp_path / "nothing")
    with pytest.raises(FileNotFoundError, match="pseudo-label folder"):
        UnlabeledDataset(cfg, mini.spec)
    cfg.pseudo_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="none of the"):
        UnlabeledDataset(cfg, mini.spec)
    cfg2 = copy.deepcopy(mini.cfg)
    cfg2.unlabeled.num_queries = 10_000
    with pytest.raises(ValueError, match="fewer than K"):
        UnlabeledDataset(cfg2, mini.spec, allow_partial=True)


# ----------------------------------------------------------------------------------------------
# EvalDataset
# ----------------------------------------------------------------------------------------------
def test_eval_dataset_samples(mini):
    ds = EvalDataset(mini.cfg, mini.spec, [C.SET_SEMANTIC, C.SET_CONDITIONAL], "val")
    expected = len(load_instruction_records(mini.cfg, C.SET_SEMANTIC, "val")) + len(load_instruction_records(mini.cfg, C.SET_CONDITIONAL, "val"))
    assert len(ds) == expected > 0
    by_id = records_by_id(mini.cfg, "val")
    kinds = set()
    for index in range(len(ds)):
        sample = ds[index]
        assert sample.kind == "eval" and sample.image_id in mini.val and len(sample.conversations) == 1
        conv = sample.conversations[0]
        record = by_id[conv.record_id]
        kinds.add(conv.set_name)
        assert conv.answer == EVAL_ANSWER == "It is <SSEG>." and conv.query == record.query  # stored query, no redraw
        assert conv.mask.dtype == torch.uint8 and set(conv.mask.unique().tolist()) <= {0, 1} and conv.mask.sum() > 0
        assert (conv.ref_pixels is not None) == (record.set == C.SET_CONDITIONAL)
    assert kinds == {C.SET_SEMANTIC, C.SET_CONDITIONAL, "conditional_templated"}
    assert torch.equal(ds[0].conversations[0].mask, ds[0].conversations[0].mask)  # deterministic
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_eval_ignore_void_marks_void_pixels(mini):
    cfg = copy.deepcopy(mini.cfg)
    cfg.data.eval_ignore_void = True
    ds = EvalDataset(cfg, mini.spec, [C.SET_SEMANTIC], "val")
    sample = ds[0]
    label = mini.adapter.load_label(sample.image_id)
    mask = sample.conversations[0].mask.numpy()
    assert set(np.unique(mask).tolist()) <= {0, 1, 255}
    assert np.array_equal(mask == 255, label == 255) and (label == 255).any()


def test_eval_dataset_errors(mini, tmp_path):
    with pytest.raises(ValueError, match="sets"):
        EvalDataset(mini.cfg, mini.spec, ["unlabeled"], "val")
    cfg = copy.deepcopy(mini.cfg)
    cfg.paths.data_dir = str(tmp_path / "none")
    with pytest.raises(FileNotFoundError, match=r"val[/\\]attribute\.jsonl"):
        EvalDataset(cfg, mini.spec, [C.SET_ATTRIBUTE], "val")


def test_eval_reports_rule_based_conditional_records_as_their_own_group(mini):
    ds = EvalDataset(mini.cfg, mini.spec, [C.SET_CONDITIONAL], "val")
    by_id = records_by_id(mini.cfg, "val")
    groups = {}
    for index in range(len(ds)):
        conv = ds[index].conversations[0]
        groups.setdefault(conv.set_name, set()).add(by_id[conv.record_id].meta["generator"])
    assert groups == {C.SET_CONDITIONAL: {"offline"}, "conditional_templated": {"rule"}}
    assert eval_group(by_id[ds.records[0].id]) == C.SET_CONDITIONAL
    templated = [r for r in ds.records if r.meta.get("style") == "templated"]
    assert templated and all(eval_group(r) == "conditional_templated" for r in templated)
    # training keeps the plain set name
    train = LabeledDataset(mini.cfg, stage([C.SET_CONDITIONAL]), mini.spec)
    assert {c.set_name for i in range(8) for c in train[i].conversations} == {C.SET_CONDITIONAL}


# ----------------------------------------------------------------------------------------------
# Conditional style mix
# ----------------------------------------------------------------------------------------------
def style_counts(ds, n=60):
    by_id = records_by_id(ds.cfg)
    counts = {"llm": 0, "templated": 0}
    for i in range(n):
        for conv in ds[i].conversations:
            counts[record_style(by_id[conv.record_id])] += 1
    return counts


def test_conditional_styles_are_drawn_one_to_one_regardless_of_record_counts(mini):
    st = stage([C.SET_CONDITIONAL])
    cfg = copy.deepcopy(mini.cfg)
    cfg.data.queries_per_image = 2
    by_style = {"llm": 0, "templated": 0}
    for record in load_instruction_records(cfg, C.SET_CONDITIONAL, "train"):
        by_style[record_style(record)] += 1
    counts = style_counts(LabeledDataset(cfg, st, mini.spec, seed=1), n=80)
    total = sum(counts.values())
    assert min(by_style.values()) > 0 and total > 0
    # drawing records uniformly would give each style its share of the file; 1:1 gives about one half each
    assert 0.3 < counts["templated"] / total < 0.7


def test_conditional_style_weights_select_a_source(mini):
    st = stage([C.SET_CONDITIONAL])
    only_llm = style_counts(LabeledDataset(mini.cfg, st, mini.spec, conditional_style_weights={"llm": 1.0, "templated": 0.0}), n=20)
    only_rule = style_counts(LabeledDataset(mini.cfg, st, mini.spec, conditional_style_weights={"templated": 1.0}), n=20)
    assert only_llm["templated"] == 0 and only_llm["llm"] > 0
    assert only_rule["llm"] == 0 and only_rule["templated"] > 0


def test_conditional_sources_option_selects_the_sources(mini):
    st = stage([C.SET_CONDITIONAL])
    counts = {}
    for sources in ("both", "llm", "templated"):
        cfg = copy.deepcopy(mini.cfg)
        cfg.data.conditional_sources = sources
        counts[sources] = style_counts(LabeledDataset(cfg, st, mini.spec, seed=2), n=30)
    assert counts["both"]["llm"] > 0 and counts["both"]["templated"] > 0  # the default mixes the two sources
    assert counts["llm"]["templated"] == 0 and counts["llm"]["llm"] > 0
    assert counts["templated"]["llm"] == 0 and counts["templated"]["templated"] > 0
    assert mini.cfg.data.conditional_sources == "both"
    bad = copy.deepcopy(mini.cfg)
    bad.data.conditional_sources = "neither"
    with pytest.raises(ValueError, match="conditional_sources"):
        LabeledDataset(bad, st, mini.spec)


def test_conditional_records_of_one_image_are_distinct(mini):
    ds = LabeledDataset(mini.cfg, stage([C.SET_CONDITIONAL]), mini.spec, seed=3)
    for i in range(20):
        ids = [c.record_id for c in ds[i].conversations]
        assert len(ids) == len(set(ids))


def test_conditional_style_weights_are_validated(mini):
    st = stage([C.SET_CONDITIONAL])
    for bad in ({"llm": 0.0, "templated": 0.0}, {"llm": -1.0}, {"other": 1.0}):
        with pytest.raises(ValueError, match="conditional_style_weights"):
            LabeledDataset(mini.cfg, st, mini.spec, conditional_style_weights=bad)
