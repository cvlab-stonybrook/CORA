"""Tests of cora.data.templates: LISA's template lists and the class phrases (Sec. 3.4.2)."""
import random

import pytest

from cora import constants as C
from cora.data import templates as T

# LISA utils/utils.py L14-42, written out independently of the module under test.
LISA_SHORT = [
    "Can you segment the {class_name} in this image?",
    "Please segment the {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",
]
LISA_LONG = ["{sent} Please respond with segmentation mask.", "{sent} Please output segmentation mask."]
LISA_ANSWERS = ["It is [SEG].", "Sure, [SEG].", "Sure, it is [SEG].", "Sure, the segmentation result is [SEG].", "[SEG]."]


def test_lists_are_lisas_without_image_token_and_with_sseg():
    assert T.SHORT_QUESTION_LIST == LISA_SHORT
    assert T.LONG_QUESTION_LIST == LISA_LONG
    assert T.ANSWER_LIST == [a.replace("[SEG]", C.SEG_TOKEN) for a in LISA_ANSWERS]
    for text in T.SHORT_QUESTION_LIST + T.LONG_QUESTION_LIST + T.ANSWER_LIST:
        assert C.IMAGE_TOKEN not in text and "[SEG]" not in text
    for answer in T.ANSWER_LIST:
        assert answer.count(C.SEG_TOKEN) == 1


def test_semantic_query_lowercases_and_covers_every_template():
    rng = random.Random(0)
    seen = {T.semantic_query("Traffic Light", rng) for _ in range(200)}
    assert seen == {t.format(class_name="traffic light") for t in LISA_SHORT}


def test_semantic_query_is_seeded():
    assert [T.semantic_query("car", random.Random(5)) for _ in range(3)] == [T.semantic_query("car", random.Random(5)) for _ in range(3)]


def test_long_query_and_random_answer():
    rng = random.Random(1)
    assert {T.long_query("The vehicle used for commuting.", rng) for _ in range(100)} == {
        t.format(sent="The vehicle used for commuting.") for t in LISA_LONG
    }
    assert {T.random_answer(rng) for _ in range(300)} == set(T.ANSWER_LIST)


@pytest.mark.parametrize("bad", ["", "   ", f"the {C.SEG_TOKEN}", f"left of {C.REF_TOKEN}", C.IMAGE_TOKEN])
def test_builders_reject_empty_text_and_special_tokens(bad):
    rng = random.Random(0)
    with pytest.raises(ValueError):
        T.semantic_query(bad, rng)
    with pytest.raises(ValueError):
        T.long_query(bad, rng)
    with pytest.raises(ValueError):
        T.wrap_description(bad, rng)


def test_wrap_description_drops_article_and_final_stop():
    rng = random.Random(0)
    queries = {T.wrap_description("The four wheeled vehicle used for personal transport.", rng) for _ in range(100)}
    assert queries == {t.format(class_name="four wheeled vehicle used for personal transport") for t in LISA_SHORT}
    assert T.wrap_description("A tall thin pole", random.Random(0)).count("tall thin pole") == 1
    assert "HTTP" in T.wrap_description("HTTP signage", random.Random(0))  # acronyms keep their case
    with pytest.raises(ValueError, match="no noun phrase"):
        T.wrap_description("the .", random.Random(0))


def test_cityscapes_phrases_follow_the_class_table():
    for class_id, name in enumerate(C.CITYSCAPES_CLASSES):
        assert T.class_phrases("cityscapes", class_id) == [name]
        assert T.singular_phrase("cityscapes", class_id) == name
        assert len(T.plural_phrases("cityscapes", class_id)) == 1
    assert T.plural_phrases("cityscapes", 13) == ["cars"]
    assert T.plural_phrases("cityscapes", 11) == ["people"]
    assert T.plural_phrases("cityscapes", 15) == ["buses"]
    assert T.is_mass_phrase("cityscapes", 8) and not T.is_mass_phrase("cityscapes", 13)
    assert T.singular_phrases("cityscapes", 13) == ["car"]


def test_phrase_lookups_reject_bad_ids_and_datasets():
    with pytest.raises(ValueError, match="outside 0..18"):
        T.class_phrases("cityscapes", 19)
    with pytest.raises(ValueError, match="outside"):
        T.singular_phrases("cityscapes", 19)
    with pytest.raises(ValueError, match="outside 0..18"):
        T.singular_phrase("cityscapes", -1)
    with pytest.raises(ValueError, match="unknown dataset"):
        T.class_phrases("coco", 0)


def test_returned_phrase_lists_are_copies():
    T.class_phrases("cityscapes", 13).append("x")
    assert T.class_phrases("cityscapes", 13) == ["car"]
