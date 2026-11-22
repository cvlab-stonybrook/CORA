"""Tests of the rule-based instruction sets and the generate_rule_sets CLI.

Synthetic label maps run through ``extract_objects`` stand in for Cityscapes; one test at the end
uses two real Cityscapes validation images (marker ``data``).
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from cora import constants as C
from cora.config import load_config
from cora.data import templates
from cora.data.objects import ensure_objects, extract_objects
from cora.data.query_db import QueryDB
from cora.data.relations import ALL_RELATIONS, anchor_groups, merge_regions, related_regions, relation_holds
from cora.data.schema import InstructionRecord, ObjectRef, iter_jsonl, read_jsonl, write_jsonl
from cora.instructions.offline import (
    ATTRIBUTE_PHRASES, OfflineRefusal, build_offline_records, refuse_if_llm_generated,
)
from cora.instructions.semantic import build_semantic_records
from cora.instructions.templated_conditional import (
    MIN_ANCHOR_FRAC, MIN_TARGET_FRAC, Candidate, area_threshold, build_templated_conditional, render_query,
)

REPO = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------------------------
class FakeAdapter:
    """In-memory dataset adapter."""

    root = Path("/nonexistent")

    def __init__(self, name, labels):
        self.name = name
        self.class_names = C.class_names(name)
        self._labels = labels

    def image_ids(self, split):
        return sorted(i for i in self._labels if i.startswith(split + "/"))

    def partition(self, partition, which):
        return sorted(self._labels)

    def image_path(self, image_id):
        return Path(image_id)

    def load_image(self, image_id):
        label = self._labels[image_id]
        image = np.zeros(label.shape + (3,), dtype=np.uint8)
        image[...] = np.linspace(40, 200, label.shape[1]).astype(np.uint8)[None, :, None]
        return image

    def load_label(self, image_id):
        return self._labels[image_id].copy()


def city_label(shift=0):
    label = np.full((96, 192), 255, dtype=np.uint8)
    label[0:10] = 10  # sky
    label[10:41, 0:60] = 2  # building
    label[70:96] = 0  # road
    label[50:66, 10 + shift:30 + shift] = 13  # three cars
    label[50:66, 70:95] = 13
    label[50:66, 150:180] = 13
    label[48:67, 110:118] = 11  # person
    return label


def city_adapter(split="train"):
    return FakeAdapter("cityscapes", {f"{split}/c/img{i}": city_label(i * 3) for i in range(4)})


@pytest.fixture
def cfg(tmp_path):
    return load_config(None, [f"paths.data_dir={tmp_path / 'data'}"])


def sorted_lines(records):
    return [r.to_json() for r in records]


# ----------------------------------------------------------------------------------------------
# Semantic set
# ----------------------------------------------------------------------------------------------
def test_semantic_records_one_per_class_present(cfg):
    adapter = city_adapter()
    ids = adapter.image_ids("train")
    records = list(build_semantic_records(cfg, adapter, "train", ids))
    per_image = {i: sorted(r.target.label_id for r in records if r.image_id == i) for i in ids}
    assert all(v == [0, 2, 10, 11, 13] for v in per_image.values())
    for record in records:
        record.validate()
        assert record.set == C.SET_SEMANTIC and record.target.is_class_level and record.anchor is None
        assert record.meta["class_level"] is True
        name = record.target.label_name
        assert any(record.query == t.format(class_name=name) for t in templates.SHORT_QUESTION_LIST)
    assert len({r.id for r in records}) == len(records)


def test_semantic_records_are_deterministic_and_order_independent(cfg):
    adapter = city_adapter()
    ids = adapter.image_ids("train")
    first = sorted_lines(build_semantic_records(cfg, adapter, "train", ids))
    assert first == sorted_lines(build_semantic_records(cfg, adapter, "train", ids))
    one_by_one = [line for i in reversed(ids) for line in sorted_lines(build_semantic_records(cfg, adapter, "train", [i]))]
    assert sorted(first) == sorted(one_by_one)
    other = load_config(None, [f"paths.data_dir={cfg.paths.data_dir}", "rule_sets.seed=7"])
    assert first != sorted_lines(build_semantic_records(other, adapter, "train", ids))


def test_semantic_min_pixels_and_errors(cfg):
    adapter = city_adapter()
    records = list(build_semantic_records(cfg, adapter, "train", adapter.image_ids("train")[:1], min_pixels=1000))
    assert sorted(r.target.label_id for r in records) == [0, 2, 10, 13]  # the 152-pixel person is dropped
    with pytest.raises(ValueError, match="min_pixels"):
        list(build_semantic_records(cfg, adapter, "train", ["train/c/img0"], min_pixels=0))
    bad = FakeAdapter("cityscapes", {"train/x": np.full((4, 4), 30, dtype=np.uint8)})
    with pytest.raises(ValueError, match="outside the 19 classes"):
        list(build_semantic_records(cfg, bad, "train", ["train/x"]))


# ----------------------------------------------------------------------------------------------
# Templated conditional set
# ----------------------------------------------------------------------------------------------
def check_templated_record(record, objects, **thresholds):
    record.validate()
    assert record.set == C.SET_CONDITIONAL and record.meta["style"] == "templated" and record.meta["generator"] == "rule"
    assert record.query.count(C.REF_TOKEN) == 1 and record.relation in ALL_RELATIONS
    members = [objects.get(i) for i in record.anchor.region_ids]
    assert len(members) == record.meta["num_anchor_regions"] and 1 <= len(members) <= 2
    assert {m.label_id for m in members} == {record.anchor.label_id}
    anchor = members[0] if len(members) == 1 else merge_regions(members)
    assert anchor.area >= area_threshold(objects, MIN_ANCHOR_FRAC)
    min_target = area_threshold(objects, MIN_TARGET_FRAC)
    assert record.target.region_ids and not set(record.anchor.region_ids) & set(record.target.region_ids)
    targets = [objects.get(i) for i in record.target.region_ids]
    assert {t.label_id for t in targets} == {record.target.label_id}
    assert all(t.area >= min_target for t in targets)
    # the target is exactly the union of the class's regions that satisfy the relation (at least one)
    expected = related_regions(objects, anchor, record.target.label_id, record.relation, min_area=min_target, **thresholds)
    assert [t.region_id for t in targets] == [r.region_id for r in expected]
    for region in targets:
        assert relation_holds(region, anchor, record.relation, (objects.height, objects.width), min_area=min_target, **thresholds)
    assert not (objects.mask(record.target.region_ids) & objects.mask(record.anchor.region_ids)).any()


def test_templated_records_follow_the_relation_rules(cfg):
    adapter = city_adapter()
    ids = adapter.image_ids("train")
    records = list(build_templated_conditional(cfg, adapter, "train", ids))
    assert records
    for record in records:
        check_templated_record(record, ensure_objects(cfg, adapter, "train", record.image_id))
    assert len({r.id for r in records}) == len(records)
    assert max(sum(r.image_id == i for r in records) for i in ids) <= cfg.rule_sets.templated_conditional_per_image


def test_templated_direction_records_cover_all_instances_of_the_class(cfg):
    """'the cars on the left of <REF>' selects every car left of the anchor, not one of them."""
    adapter = city_adapter()
    objects = ensure_objects(cfg, adapter, "train", "train/c/img0")
    person = objects.of_class(11)[0].region_id
    found = None
    for seed in range(60):
        seeded = load_config(None, [f"paths.data_dir={cfg.paths.data_dir}", f"rule_sets.seed={seed}"])
        for record in build_templated_conditional(seeded, adapter, "train", ["train/c/img0"], per_image=50, relations=["to the left of"]):
            if record.anchor.region_ids == [person] and record.target.label_id == 13:
                found = record
        if found:
            break
    assert found is not None and len(found.target.region_ids) == 2  # two of the three cars are left of the person
    assert "cars" in found.query and C.REF_TOKEN in found.query and "left of" in found.query


def test_templated_is_deterministic_seeded_and_order_independent(cfg):
    adapter = city_adapter()
    ids = adapter.image_ids("train")
    first = sorted_lines(build_templated_conditional(cfg, adapter, "train", ids))
    assert first == sorted_lines(build_templated_conditional(cfg, adapter, "train", ids))
    per_image = [l for i in reversed(ids) for l in sorted_lines(build_templated_conditional(cfg, adapter, "train", [i]))]
    assert sorted(first) == sorted(per_image)
    other = load_config(None, [f"paths.data_dir={cfg.paths.data_dir}", "rule_sets.seed=1"])
    assert first != sorted_lines(build_templated_conditional(other, adapter, "train", ids))


def test_templated_per_image_cap_and_distinct_anchors_first(cfg):
    adapter = city_adapter()
    for record_count in (1, 3):
        records = list(build_templated_conditional(cfg, adapter, "train", ["train/c/img1"], per_image=record_count))
        assert len(records) == record_count
        assert len({tuple(r.anchor.region_ids) for r in records}) == record_count
    assert list(build_templated_conditional(cfg, adapter, "train", ["train/c/img1"], per_image=0)) == []
    many = list(build_templated_conditional(cfg, adapter, "train", ["train/c/img1"], per_image=10_000))
    assert len(many) == len({r.id for r in many}) > 20


def test_templated_respects_relation_subset_and_min_areas(cfg):
    adapter = city_adapter()
    only_near = list(build_templated_conditional(cfg, adapter, "train", adapter.image_ids("train"), relations=["near"], per_image=50))
    assert only_near and {r.relation for r in only_near} == {"near"}
    strict = list(build_templated_conditional(cfg, adapter, "train", ["train/c/img0"], per_image=50, min_anchor_frac=0.05))
    objects = ensure_objects(cfg, adapter, "train", "train/c/img0")
    assert strict and all(sum(objects.get(i).area for i in r.anchor.region_ids) >= 0.05 * 96 * 192 for r in strict)
    none = list(build_templated_conditional(cfg, adapter, "train", ["train/c/img0"], per_image=50, min_target_frac=0.9))
    assert none == []


def test_templated_query_wording_and_anchor_name(cfg):
    adapter = city_adapter()
    omit = list(build_templated_conditional(cfg, adapter, "train", adapter.image_ids("train"), per_image=50))
    include = list(build_templated_conditional(cfg, adapter, "train", adapter.image_ids("train"), per_image=50, anchor_name="include"))
    assert len(omit) == len(include)
    assert all(f"the {r.anchor.label_name} {C.REF_TOKEN}" in q.query for r, q in zip(omit, include))
    assert not any(f"the {r.anchor.label_name} {C.REF_TOKEN}" in r.query for r in omit)
    starts = {r.query.split()[0] for r in omit}
    assert {"Segment", "Can", "Please", "Find"} <= starts  # several wordings
    assert all(r.query[-1] in ".?" and r.query[0].isupper() for r in omit)
    assert any(r.query.startswith("Segment the cars ") for r in omit)  # Fig. 2 wording
    assert {r.meta["anchor_name"] for r in include} == {"include"}


def test_templated_rejects_bad_options(cfg):
    adapter = city_adapter()
    with pytest.raises(ValueError, match="anchor_name"):
        list(build_templated_conditional(cfg, adapter, "train", ["train/c/img0"], anchor_name="maybe"))
    with pytest.raises(ValueError, match="unknown relation"):
        list(build_templated_conditional(cfg, adapter, "train", ["train/c/img0"], relations=["beside"]))
    with pytest.raises(ValueError, match="per_image"):
        list(build_templated_conditional(cfg, adapter, "train", ["train/c/img0"], per_image=-1))
    with pytest.raises(ValueError, match="unknown relation threshold"):
        list(build_templated_conditional(cfg, adapter, "train", ["train/c/img0"], bogus=1))


def test_templated_mass_class_grammar():
    objs = extract_objects("x", city_label(), C.CITYSCAPES_CLASSES)
    candidate = Candidate(objs.get("11_1"), 10, "above", (objs.get("10_1"),))
    queries = {render_query("cityscapes", candidate, random.Random(s))[0] for s in range(80)}
    assert "Segment all the sky that is above <REF>." in queries
    assert "Segment the sky above <REF>." in queries
    plural = Candidate(objs.get("11_1"), 13, "near", tuple(objs.of_class(13)))
    assert "Segment all the cars that are close to <REF>." in {render_query("cityscapes", plural, random.Random(s))[0] for s in range(200)}
    with pytest.raises(ValueError, match="anchor_name"):
        render_query("cityscapes", candidate, random.Random(0), "nope")


def test_templated_query_number_follows_the_number_of_target_regions():
    objs = extract_objects("x", city_label(), C.CITYSCAPES_CLASSES)
    one = Candidate(objs.get("11_1"), 13, "near", (objs.of_class(13)[0],))
    many = Candidate(objs.get("11_1"), 13, "near", tuple(objs.of_class(13)))
    assert len(many.targets) > 1
    singles = {render_query("cityscapes", one, random.Random(s))[0] for s in range(200)}
    assert "Segment the car close to <REF>." in singles and "Segment the car that is close to <REF>." in singles
    assert not any("cars" in q or "all the" in q or "are" in q.split() for q in singles), singles
    plurals = {render_query("cityscapes", many, random.Random(s))[0] for s in range(200)}
    assert all("cars" in q for q in plurals)
    mass = Candidate(objs.get("11_1"), 10, "above", (objs.get("10_1"),))  # a mass noun stays a mass noun
    mass_queries = {render_query("cityscapes", mass, random.Random(s))[0] for s in range(80)}
    assert all("sky" in q for q in mass_queries) and "Segment all the sky that is above <REF>." in mass_queries
    assert "Segment the car close to the person <REF>." in {
        render_query("cityscapes", one, random.Random(s), "include")[0] for s in range(200)}


def test_templated_records_use_the_singular_for_a_single_target_region(cfg):
    adapter = city_adapter()
    records = list(build_templated_conditional(cfg, adapter, "train", adapter.image_ids("train"), per_image=50))
    singles = [r for r in records if len(r.target.region_ids) == 1 and not templates.is_mass_phrase("cityscapes", r.target.label_id)]
    assert singles
    for record in singles:
        assert templates.singular_phrase("cityscapes", record.target.label_id) in record.query
        assert templates.plural_phrases("cityscapes", record.target.label_id)[0] not in record.query


# ----------------------------------------------------------------------------------------------
# Offline stand-in
# ----------------------------------------------------------------------------------------------
def test_offline_records_are_marked_and_valid(cfg):
    adapter = city_adapter()
    ids = adapter.image_ids("train")
    records = list(build_offline_records(cfg, adapter, "train", ids, ["attribute", "conditional"]))
    assert {r.set for r in records} == {"attribute", "conditional"}
    for record in records:
        record.validate()
        assert record.meta["generator"] == "offline"
        assert record.description in ATTRIBUTE_PHRASES["cityscapes"][record.target.label_id]
        assert record.description in record.query and record.target.region_ids
        if record.set == "attribute":
            assert record.anchor is None and C.REF_TOKEN not in record.query
        else:
            assert record.query.count(C.REF_TOKEN) == 1 and record.anchor is not None
    per_image = cfg.rule_sets.attribute_targets_per_image * cfg.rule_sets.qa_per_target
    assert all(sum(r.image_id == i and r.set == "attribute" for r in records) <= per_image for i in ids)
    assert all(sum(r.image_id == i and r.set == "conditional" for r in records) <= cfg.rule_sets.templated_conditional_per_image for i in ids)
    assert records == list(build_offline_records(cfg, adapter, "train", ids, ["attribute", "conditional"]))


def test_offline_conditional_geometry_matches_relations(cfg):
    adapter = city_adapter()
    for record in build_offline_records(cfg, adapter, "train", adapter.image_ids("train"), ["conditional"]):
        objects = ensure_objects(cfg, adapter, "train", record.image_id)
        anchor = objects.get(record.anchor.region_ids[0])
        expected = related_regions(objects, anchor, record.target.label_id, record.relation,
                                   min_area=area_threshold(objects, MIN_TARGET_FRAC))
        assert record.target.region_ids == [r.region_id for r in expected]


def test_offline_rejects_unsupported_sets(cfg):
    adapter = city_adapter()
    with pytest.raises(ValueError, match="subset"):
        list(build_offline_records(cfg, adapter, "train", ["train/c/img0"], ["semantic"]))
    with pytest.raises(ValueError, match="subset"):
        list(build_offline_records(cfg, adapter, "train", ["train/c/img0"], []))


def llm_record(set_name="attribute", generator="llm"):
    return InstructionRecord(
        id="llm-1", dataset="cityscapes", split="train", image_id="train/c/img0", set=set_name,
        query="Can you segment the large vehicle?", target=ObjectRef(15, "bus", ["15_1"]), meta={"generator": generator},
    )


def test_refusal_rule_protects_llm_generated_files(cfg):
    path = cfg.instructions_path("train", "attribute")
    refuse_if_llm_generated(cfg, "train", "attribute")  # nothing there yet
    write_jsonl(path, [llm_record()])
    with pytest.raises(OfflineRefusal, match="llm"):
        refuse_if_llm_generated(cfg, "train", "attribute")
    refuse_if_llm_generated(cfg, "train", "conditional")  # other set is untouched
    write_jsonl(path, [llm_record(generator="offline")])
    refuse_if_llm_generated(cfg, "train", "attribute")  # earlier offline output may be replaced
    no_generator = llm_record()
    no_generator.meta = {}
    write_jsonl(path, [no_generator])
    with pytest.raises(OfflineRefusal):  # unknown provenance counts as LLM data
        refuse_if_llm_generated(cfg, "train", "attribute")


def test_refusal_rule_covers_variant_files_but_not_rule_based_ones(cfg):
    variant = cfg.instructions_path("train", "conditional", "templated")
    rule = llm_record("conditional", generator="rule")
    rule.anchor = ObjectRef(2, "building", ["2_1"])
    rule.query = f"Segment the buses near {C.REF_TOKEN}."
    write_jsonl(variant, [rule])
    refuse_if_llm_generated(cfg, "train", "conditional")
    llm_variant = cfg.instructions_path("train", "conditional", "llm")
    llm = llm_record("conditional")
    llm.anchor, llm.query = rule.anchor, rule.query
    write_jsonl(llm_variant, [llm])
    with pytest.raises(OfflineRefusal):
        refuse_if_llm_generated(cfg, "train", "conditional")


# ----------------------------------------------------------------------------------------------
# CLIs on a synthetic Cityscapes folder
# ----------------------------------------------------------------------------------------------
LABEL_IDS = {0: 7, 2: 11, 10: 23, 11: 24, 13: 26}  # trainId -> Cityscapes labelId


@pytest.fixture
def mini_cityscapes(tmp_path):
    root = tmp_path / "cityscapes"
    for i in range(3):
        stem = f"city_000000_00000{i}"
        label = city_label(i * 3)
        ids = np.zeros(label.shape, dtype=np.uint8)
        for train_id, label_id in LABEL_IDS.items():
            ids[label == train_id] = label_id
        image = np.full(label.shape + (3,), 90 + 20 * i, dtype=np.uint8)
        for sub, name, data in (("leftImg8bit", f"{stem}_leftImg8bit.png", image), ("gtFine", f"{stem}_gtFine_labelIds.png", ids)):
            folder = root / sub / "val" / "city"
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / name), data)
    return root


def cli_args(tmp_path, root, *rest):
    return ["--set", f"paths.cityscapes_root={root}", "--set", f"paths.data_dir={tmp_path / 'data'}", *rest]


def test_cli_semantic_templated_and_offline(tmp_path, mini_cityscapes, capsys):
    tool = load_tool("generate_rule_sets")
    assert tool.main(cli_args(tmp_path, mini_cityscapes, "semantic", "--split", "val")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["records"] == 15 and report["images"] == 3
    semantic = read_jsonl(Path(report["path"]))
    assert {r.target.label_name for r in semantic} == {"road", "building", "sky", "person", "car"}

    assert tool.main(cli_args(tmp_path, mini_cityscapes, "templated-conditional", "--split", "val", "--per-image", "2")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["path"].endswith("instructions/val/conditional_templated.jsonl") and report["records"] == 6
    assert sum(report["per_relation"].values()) == 6
    assert all(r.meta["style"] == "templated" for r in read_jsonl(Path(report["path"])))

    assert tool.main(cli_args(tmp_path, mini_cityscapes, "offline", "--split", "val", "--limit", "2", "--sets", "attribute", "conditional")) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"attribute", "conditional"} and report["attribute"]["records"] > 0
    assert all(r.meta["generator"] == "offline" for r in iter_jsonl(report["attribute"]["path"]))


def test_cli_offline_refuses_to_mix_with_llm_files(tmp_path, mini_cityscapes, capsys):
    tool = load_tool("generate_rule_sets")
    cfg = load_config(None, [f"paths.data_dir={tmp_path / 'data'}"])
    llm = llm_record("conditional")
    llm.anchor, llm.query = ObjectRef(2, "building", ["2_1"]), f"Segment the bus near {C.REF_TOKEN}."
    write_jsonl(cfg.instructions_path("val", "conditional"), [llm])
    before = cfg.instructions_path("val", "conditional").read_text()
    assert tool.main(cli_args(tmp_path, mini_cityscapes, "offline", "--split", "val", "--sets", "conditional")) == 2
    assert "refused" in capsys.readouterr().err
    assert cfg.instructions_path("val", "conditional").read_text() == before
    assert not cfg.instructions_path("val", "attribute").exists()


def test_cli_query_db(tmp_path, mini_cityscapes, capsys):
    tool = load_tool("generate_rule_sets")
    cfg = load_config(None, [f"paths.data_dir={tmp_path / 'data'}"])
    assert tool.main(cli_args(tmp_path, mini_cityscapes, "query-db")) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["queries_per_class"]) == 19 and QueryDB.load(cfg.query_db_path).problems() == []
    with pytest.raises(SystemExit, match="already exists"):
        tool.main(cli_args(tmp_path, mini_cityscapes, "query-db"))
    assert tool.main(cli_args(tmp_path, mini_cityscapes, "query-db", "--force")) == 0
    capsys.readouterr()
    assert tool.main(cli_args(tmp_path, mini_cityscapes, "query-db", "--dry-run")) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["dry_run"] and dry["classes"] == 19


def test_cli_rejects_empty_selection(tmp_path, mini_cityscapes):
    tool = load_tool("generate_rule_sets")
    (tmp_path / "empty.txt").write_text("")
    with pytest.raises(SystemExit, match="no images selected"):
        tool.main(cli_args(tmp_path, mini_cityscapes, "semantic", "--split", "val", "--image-list", str(tmp_path / "empty.txt")))


def test_cli_drops_repeated_image_ids(tmp_path, mini_cityscapes, capsys):
    tool = load_tool("generate_rule_sets")
    assert tool.main(cli_args(tmp_path, mini_cityscapes, "semantic", "--split", "val")) == 0
    report = json.loads(capsys.readouterr().out)
    ids = sorted({r.image_id for r in read_jsonl(Path(report["path"]))})
    listing = tmp_path / "ids.txt"
    listing.write_text("\n".join([ids[0], ids[1], ids[0]]) + "\n")
    assert tool.main(cli_args(tmp_path, mini_cityscapes, "semantic", "--split", "val", "--image-list", str(listing))) == 0
    report = json.loads(capsys.readouterr().out)
    records = read_jsonl(Path(report["path"]))
    assert report["images"] == 2 and len({r.id for r in records}) == len(records)


def test_generate_rule_sets_usage_examples_parse():
    gen = load_tool("generate_rule_sets")
    examples = [line.split("generate_rule_sets.py", 1)[1].split() for line in gen.__doc__.splitlines()
                if line.strip().startswith("python tools/generate_rule_sets.py")]
    assert len(examples) >= 5
    for argv in examples:
        gen.parse_args(argv)  # a documented command line must be accepted
    with pytest.raises(SystemExit):  # no abbreviation: "--set" must not silently become "--sets"
        gen.parse_args(["offline", "--split", "train", "--set", "data.partition=1_30"])


# ----------------------------------------------------------------------------------------------
# Real Cityscapes images
# ----------------------------------------------------------------------------------------------
@pytest.mark.data
def test_real_cityscapes_templated_conditional_records_are_consistent(cityscapes_root, tmp_path):
    from cora.data.adapters import get_adapter

    cfg = load_config(None, [f"paths.data_dir={tmp_path / 'data'}", f"paths.cityscapes_root={cityscapes_root}"])
    adapter = get_adapter(cfg)
    ids = adapter.image_ids("val")[:2]
    assert len(ids) == 2
    records = list(build_templated_conditional(cfg, adapter, "val", ids))
    assert {r.image_id for r in records} == set(ids)  # both images yield records
    for record in records:
        objects = ensure_objects(cfg, adapter, "val", record.image_id)
        check_templated_record(record, objects)
        target_mask = objects.mask(record.target.region_ids)
        anchor_mask = objects.mask(record.anchor.region_ids)
        assert target_mask.any() and anchor_mask.any() and not (target_mask & anchor_mask).any()
        label = adapter.load_label(record.image_id)
        assert (label[target_mask] == record.target.label_id).all()  # the mask is the class's pixels
        assert (label[anchor_mask] == record.anchor.label_id).all()
        members = [objects.get(i) for i in record.anchor.region_ids]
        anchor = members[0] if len(members) == 1 else merge_regions(members)
        if record.relation in ("to the left of", "to the right of"):
            sign = -1 if record.relation == "to the left of" else 1
            for region in map(objects.get, record.target.region_ids):
                assert sign * (region.centroid[0] - anchor.centroid[0]) > 0
        if record.relation in ("above", "below"):
            sign = -1 if record.relation == "above" else 1
            for region in map(objects.get, record.target.region_ids):
                assert sign * (region.centroid[1] - anchor.centroid[1]) > 0
        if record.relation == "near":
            for region in map(objects.get, record.target.region_ids):
                assert relation_holds(region, anchor, "near", (objects.height, objects.width),
                                      target_mask=objects.mask([region.region_id]), anchor_mask=anchor_mask)
    semantic = list(build_semantic_records(cfg, adapter, "val", ids))
    assert semantic and all(r.target.label_id in range(19) for r in semantic)


# ----------------------------------------------------------------------------------------------
# Depth relations, group anchors, singular phrases (supplement Table 5, Fig. 2)
# ----------------------------------------------------------------------------------------------
def depth_group_label():
    label = np.full((96, 192), 255, dtype=np.uint8)
    label[50:70, 20:30] = 11  # two neighbouring pedestrians: a group anchor
    label[50:70, 34:44] = 11
    label[30:45, 100:130] = 13  # a far car ...
    label[60:85, 95:140] = 13  # ... and a near one on the same line of sight
    label[0:10] = 10
    return label


def depth_group_adapter():
    return FakeAdapter("cityscapes", {"train/d/img0": depth_group_label()})


def test_templated_depth_relations_and_group_anchors_on_cityscapes(cfg):
    adapter = depth_group_adapter()
    records = list(build_templated_conditional(cfg, adapter, "train", ["train/d/img0"], per_image=500))
    objects = ensure_objects(cfg, adapter, "train", "train/d/img0")
    assert {"in front of", "behind"} <= {r.relation for r in records}
    groups = [r for r in records if len(r.anchor.region_ids) == 2]
    assert groups and {tuple(r.anchor.region_ids) for r in groups} == {("11_1", "11_2")}
    assert all(r.meta["num_anchor_regions"] == 2 for r in groups)
    for record in records:
        check_templated_record(record, objects)
    depth = [r for r in records if r.relation in ("in front of", "behind")]
    assert depth and all(r.target.label_name == "car" and r.anchor.label_name == "car" for r in depth)
    assert all(("in front of" in r.query) or ("behind" in r.query) or ("at the back of" in r.query) for r in depth)
    front = next(r for r in depth if r.relation == "in front of")
    assert (front.anchor.region_ids, front.target.region_ids) == (["13_1"], ["13_2"])  # far car is 13_1 in raster order


def test_templated_group_anchors_can_be_switched_off(cfg):
    adapter = depth_group_adapter()
    single = list(build_templated_conditional(cfg, adapter, "train", ["train/d/img0"], per_image=500, use_anchor_groups=False))
    assert single and all(len(r.anchor.region_ids) == 1 for r in single)
    with pytest.raises(ValueError, match="unknown relation"):
        list(build_templated_conditional(cfg, adapter, "train", ["train/d/img0"], relations=["in back of"]))


def test_group_anchor_with_include_names_the_plural_class():
    objs = extract_objects("x", depth_group_label(), C.CITYSCAPES_CLASSES)
    group = merge_regions(objs.of_class(11))
    candidate = Candidate(group, 13, "near", (objs.of_class(13)[0],))
    queries = {render_query("cityscapes", candidate, random.Random(s), "include")[0] for s in range(40)}
    assert all("the people <REF>" in q for q in queries)
    assert anchor_groups(objs)[0].region_id == group.region_id


# ----------------------------------------------------------------------------------------------
# Instruction statistics (supplement Sec. 7.1, Fig. 2a / 2b)
# ----------------------------------------------------------------------------------------------
def stats_records():
    def record(i, label, query, question=None):
        meta = {"question": question} if question else {}
        return InstructionRecord(id=str(i), dataset="cityscapes", split="train", image_id="train/x", set="conditional",
                                 query=query, target=ObjectRef(13, label, ["13_1"]), anchor=ObjectRef(11, "person", ["11_1"]),
                                 meta=meta)
    return [
        record(0, "car", "Segment the cars near <REF>."),
        record(1, "car", "Segment the cars on the left of <REF>.", "Segment the cars on the left of anchor region 11_2."),
        record(2, "pole", "Segment the poles above <REF>."),
        record(3, "car", "Segment the car behind <REF>."),
    ]


def test_instruction_stats_counts_lengths_and_order():
    from cora.instructions.stats import instruction_stats, paper_text, stats_dict, stats_table

    records = stats_records()
    stats = instruction_stats(records)
    assert [s.label_name for s in stats] == ["car", "pole"] and [s.count for s in stats] == [3, 1]
    car = stats[0]
    assert car.mean_words == pytest.approx((5 + 8 + 5) / 3) and car.mean_chars == pytest.approx(
        sum(len(r.query) for r in records if r.target.label_name == "car") / 3)
    assert paper_text(records[1]).endswith("region 11_2.") and paper_text(records[0]) == records[0].query
    assert car.mean_words_paper == pytest.approx((5 + 10 + 5) / 3)  # the stored question counts "anchor region 11_2" (3 words)
    summary = stats_dict(stats)
    assert summary["total"] == 4 and summary["classes"][0]["label_name"] == "car"
    assert "car" in stats_table(stats) and "paper-style" in stats_table(stats)
    with pytest.raises(ValueError, match="no instruction records"):
        instruction_stats([])


def test_plot_stats_writes_both_panels(tmp_path):
    from cora.instructions.stats import instruction_stats, plot_stats

    out = plot_stats(instruction_stats(stats_records()), tmp_path / "plots" / "fig2.png")
    assert out.is_file() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="unit"):
        plot_stats(instruction_stats(stats_records()), tmp_path / "x.png", unit="bytes")


def test_cli_stats_reads_stored_files_and_writes_json_and_plot(tmp_path, mini_cityscapes, capsys):
    tool = load_tool("generate_rule_sets")
    assert tool.main(cli_args(tmp_path, mini_cityscapes, "templated-conditional", "--split", "val", "--per-image", "3")) == 0
    capsys.readouterr()
    json_path, plot_path = tmp_path / "stats.json", tmp_path / "fig2.png"
    assert tool.main(cli_args(tmp_path, mini_cityscapes, "stats", "--split", "val", "--variant", "templated",
                              "--json", str(json_path), "--plot", str(plot_path))) == 0
    out = capsys.readouterr().out
    assert "class" in out.splitlines()[0] and "all" in out
    saved = json.loads(json_path.read_text())
    assert saved["total"] > 0 and saved["classes"] == sorted(saved["classes"], key=lambda c: (-c["count"], c["label_name"]))
    assert plot_path.is_file()
    with pytest.raises(SystemExit, match="no instruction file"):
        tool.main(cli_args(tmp_path, mini_cityscapes, "stats", "--split", "train"))
