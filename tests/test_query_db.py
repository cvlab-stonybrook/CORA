"""Tests of the per-class query database D (paper Sec. 3.2.1, Fig. 3): rules, storage and the shipped files."""
import json
import random
import re

import pytest

from cora import constants as C
from cora.config import load_config
from cora.data.query_db import MIN_QUERIES, QueryDB, check_query, normalise, shipped_path

DATASETS = ("cityscapes",)


@pytest.fixture(scope="module", params=DATASETS)
def shipped(request):
    return QueryDB.load(shipped_path(request.param))


# ----------------------------------------------------------------------------------------------
# The shipped databases
# ----------------------------------------------------------------------------------------------
def test_shipped_databases_cover_every_class(shipped):
    names = C.class_names(shipped.dataset)
    assert shipped.class_ids() == list(range(len(names)))
    for class_id in shipped.class_ids():
        assert len(shipped.queries(class_id)) >= MIN_QUERIES, names[class_id]


def test_shipped_databases_obey_every_rule(shipped):
    """Count, special tokens, position words, bare class names, other class names, plural / mass phrasing."""
    assert shipped.problems() == []


def test_shipped_queries_are_unique_within_and_across_classes(shipped):
    seen = {}
    for class_id in shipped.class_ids():
        for query in shipped.queries(class_id):
            key = normalise(query)
            assert key not in seen, f"{query!r} appears for classes {seen[key]} and {class_id}"
            seen[key] = class_id


def test_shipped_queries_vary_in_length_and_wording(shipped):
    names = C.class_names(shipped.dataset)
    for class_id in shipped.class_ids():
        queries = shipped.queries(class_id)
        lengths = [len(q.split()) for q in queries]
        assert len(set(lengths)) >= 6, f"{names[class_id]}: too uniform in length"
        assert max(lengths) - min(lengths) >= 6, names[class_id]
        starts = {" ".join(q.split()[:2]).lower() for q in queries}
        assert len(starts) >= 2, f"{names[class_id]}: all queries start alike"


def test_shipped_queries_hold_no_special_token_or_position_word(shipped):
    """Explicit restatement of the query rules, independent of check_query."""
    banned = re.compile(r"\b(left|right|top|bottom|front|behind|near|next to|beside)\b|[<>\[\]]|\d", re.IGNORECASE)
    for class_id in shipped.class_ids():
        for query in shipped.queries(class_id):
            assert not banned.search(query), query
            for token in (C.IMAGE_TOKEN, C.SEG_TOKEN, C.REF_TOKEN, C.REF_MASK_TOKEN, C.REF_BOX_TOKEN):
                assert token not in query


def test_no_query_is_the_bare_class_name(shipped):
    from cora.data import templates

    for class_id in shipped.class_ids():
        forms = {normalise(p) for p in templates.class_phrases(shipped.dataset, class_id)}
        forms |= {normalise(templates.singular_phrase(shipped.dataset, class_id))}
        forms |= {normalise(p) for p in templates.plural_phrases(shipped.dataset, class_id)}
        for query in shipped.queries(class_id):
            stripped = normalise(re.sub(r"^(can you |please )?segment( all)?( the)?", "", query, flags=re.IGNORECASE))
            assert stripped not in forms, query


def test_paper_examples_are_in_the_car_class():
    db = QueryDB.load(shipped_path("cityscapes"))
    cars = db.queries(13)
    assert "Segment the sedan used for daily commute." in cars
    assert "Segment all small, low-height vehicles typically used for personal transportation." in cars


def test_every_class_can_supply_k_equal_seven_queries(shipped):
    rng = random.Random(0)
    for class_id in shipped.class_ids():
        drawn = shipped.sample(class_id, 7, rng)
        assert len(drawn) == len(set(drawn)) == 7


# ----------------------------------------------------------------------------------------------
# check_query
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query, fragment",
    [
        ("Segment the cars on the left side of the street.", "position word"),
        ("Segment the cars that are near the traffic lights.", "position word"),
        ("Segment the small vehicles in front of buildings.", "position word"),
        ("Segment the vehicles next to the curb.", "position word"),
        ("Segment two vehicles used for personal transport.", "count"),
        ("Segment the 3 vehicles used for personal transport.", "count"),
        ("Segment a vehicle used for personal transport.", "singular determiner"),
        ("Segment this vehicle used for personal transport.", "singular determiner"),
        ("Segment the vehicles <REF> used for personal transport.", "special token"),
        ("Segment the vehicles <SSEG> used for personal transport.", "special token"),
        ("Segment the vehicles used for [SEG] transport.", "special token"),
        ("Segment the car.", "class name"),
        ("Can you segment all the cars?", "class name"),
        ("Segment all the buses that carry commuters.", "names another class"),
        ("Highlight small vehicles used for personal transport.", "segmentation instruction"),
        ("   ", "empty"),
    ],
)
def test_check_query_flags_each_rule(query, fragment):
    reasons = check_query(query, "cityscapes", 13)
    assert any(fragment in r for r in reasons), reasons


@pytest.mark.parametrize(
    "dataset, class_id, query, fragment",
    [
        ("cityscapes", 11, "Segment the pedestrians with backpacks, umbrellas or prams.", "incidental attribute"),
        ("cityscapes", 12, "Segment all the cyclists wearing helmets.", "incidental attribute"),
        ("cityscapes", 3, "Segment all the graffiti-covered concrete slabs.", "incidental attribute"),
        ("cityscapes", 15, "Segment the tall yellow motor vehicles that transport dozens of travellers.", "confusable"),
        ("cityscapes", 11, "Segment every human who is walking a dog.", "incidental attribute"),
        ("cityscapes", 11, "Segment the human beings walking or standing on foot.", "confusable"),
        ("cityscapes", 14, "Please segment the cement mixers and moving vans.", "confusable"),
        ("cityscapes", 5, "Segment the steel tubes standing upright to hold signal heads and boards.", "confusable"),
        ("cityscapes", 12, "Segment the humans sitting on and steering bikes or scooters.", "confusable"),
        ("cityscapes", 15, "Segment the airport shuttles, tour coaches and city transit vehicles.", "confusable"),
        ("cityscapes", 16, "Segment all the vehicles made of linked coaches that depart from stations.", "confusable"),
        ("cityscapes", 4, "Segment the lines of rails and pickets that walkers cannot easily cross.", "confusable"),
        ("cityscapes", 18, "Segment the frame-and-spoke vehicles that run without a motor.", "confusable"),
    ],
)
def test_check_query_flags_restricting_attributes_and_confusable_words(dataset, class_id, query, fragment):
    reasons = check_query(query, dataset, class_id)
    assert any(fragment in r for r in reasons), reasons


def test_check_query_accepts_the_rewritten_class_wide_queries():
    for dataset, class_id, query in [
        ("cityscapes", 12, "Segment the humans perched on saddles, including their arms and legs."),
        ("cityscapes", 14, "Segment the vehicles and trailers built to carry heavy loads."),
        ("cityscapes", 11, "Segment the pedestrians of all ages, from children to the elderly."),
        ("cityscapes", 6, "Segment the boxes with red, amber and green lamps that tell motorists when to stop and go."),
        ("cityscapes", 8, "Segment the dense green foliage made of leaves and branches."),
    ]:
        assert check_query(query, dataset, class_id) == [], (query, check_query(query, dataset, class_id))


def test_check_query_accepts_a_good_query():
    assert check_query("Segment all small, low-height vehicles typically used for personal transportation.", "cityscapes", 13) == []
    assert check_query("Segment the sedan used for daily commute.", "cityscapes", 13) == []


# ----------------------------------------------------------------------------------------------
# QueryDB
# ----------------------------------------------------------------------------------------------
def make_db(extra=None):
    queries = {13: [f"Segment the vehicle type {chr(97 + i)} for family use." for i in range(8)]}
    queries.update(extra or {})
    return QueryDB(queries, "cityscapes", meta={"source": "test"})


def test_sample_returns_distinct_queries_and_is_seeded():
    db = make_db()
    first = db.sample(13, 7, random.Random(3))
    assert len(set(first)) == 7 and set(first) <= set(db.queries(13))
    assert first == db.sample(13, 7, random.Random(3))
    assert db.sample(13, 0, random.Random(0)) == []


def test_sample_rejects_too_many_or_unknown_class():
    db = make_db()
    with pytest.raises(ValueError, match="8 queries, cannot draw 9"):
        db.sample(13, 9, random.Random(0))
    with pytest.raises(ValueError, match="no class 3"):
        db.sample(3, 1, random.Random(0))


def test_constructor_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown dataset"):
        QueryDB({0: ["Segment x."]}, "coco")
    with pytest.raises(ValueError, match="outside 0..18"):
        QueryDB({19: ["Segment x."]}, "cityscapes")
    with pytest.raises(ValueError, match="duplicate"):
        QueryDB({0: ["Segment the tarmac.", "segment the  tarmac"]}, "cityscapes")
    with pytest.raises(ValueError, match="non-empty"):
        QueryDB({0: ["  "]}, "cityscapes")


def test_validate_reports_missing_classes_and_short_lists():
    db = make_db()
    problems = db.problems()
    assert any(p.startswith("road: no queries") for p in problems)
    assert any("8 queries, need at least 12" in p for p in problems)
    with pytest.raises(ValueError, match="invalid query database"):
        db.validate()


def test_save_load_round_trip(tmp_path):
    db = make_db()
    path = tmp_path / "sub" / "db.json"
    db.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["dataset"] == "cityscapes" and raw["classes"]["13"]["name"] == "car" and raw["meta"] == {"source": "test"}
    back = QueryDB.load(path)
    assert back.class_ids() == [13] and back.queries(13) == db.queries(13) and back.meta == {"source": "test"}


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        QueryDB.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError, match="'dataset' and 'classes'"):
        QueryDB.load(bad)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"dataset": "cityscapes", "classes": {"13": {"name": "bus", "queries": ["Segment x."]}}}))
    with pytest.raises(ValueError, match="is named 'bus'"):
        QueryDB.load(wrong)


def test_for_config_prefers_the_data_folder_over_the_shipped_file(tmp_path):
    cfg = load_config(None, [f"paths.data_dir={tmp_path}"])
    assert QueryDB.for_config(cfg).problems() == []  # shipped file
    custom = make_db()
    custom.save(cfg.query_db_path)
    assert QueryDB.for_config(cfg).class_ids() == [13]


def test_shipped_path_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        shipped_path("coco")


# ----------------------------------------------------------------------------------------------
# Wheel-count modifiers and the provenance note
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("query, class_id", [
    ("Segment the four-wheeled vehicle designed for everyday private transport.", 13),  # Fig. 5 attribute instruction
    ("Segment the four-wheeled vehicles designed for everyday private transport.", 13),
    ("Segment all the humans seated on two-wheeled vehicles, motorised or not.", 12),
    ("Segment the two-wheelers with a saddle and a chain.", 18),
])
def test_class_defining_wheel_counts_are_not_instance_counts(query, class_id):
    assert check_query(query, "cityscapes", class_id) == []


@pytest.mark.parametrize("query", [
    "Segment two vehicles used for personal transport.",
    "Segment the four-cars used for personal transport.",
    "Segment the two-door vehicles used for personal transport.",
    "Segment the 4-wheeled vehicles used for personal transport.",
])
def test_other_numbers_stay_rejected(query):
    assert any("count" in p for p in check_query(query, "cityscapes", 13))


def test_shipped_queries_are_not_subset_selecting_or_bare_names():
    db = QueryDB.load(shipped_path("cityscapes"))
    rider = " ".join(db.queries(12)).lower()
    for gone in ("couriers", "sports cyclists", "commuters who steer"):
        assert gone not in rider
    assert "retaining masonry" not in " ".join(db.queries(3)).lower()
    road = [q.lower() for q in db.queries(0)]
    assert "segment all the tarmac." not in road and not any("parking aisles" in q for q in road)
    assert "Segment the four-wheeled vehicles designed for everyday private transport." in db.queries(13)


def test_for_config_says_when_the_database_is_hand_written(tmp_path, caplog):
    cfg = load_config(None, [f"paths.data_dir={tmp_path / 'data'}"])
    with caplog.at_level("INFO", logger="cora.data.query_db"):
        db = QueryDB.for_config(cfg)
    assert db.meta["source"] == "hand-written"
    assert any("hand-written" in r.getMessage() for r in caplog.records)
