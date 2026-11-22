"""Question / answer templates and class phrases (paper Sec. 3.4.2).

The semantic instruction set "fit[s] class labels into segmentation query templates" (Sec. 3.4.2); the
template lists are LISA's (``utils/utils.py``, Apache-2.0), copied verbatim except that

* the leading ``<image>\\n`` is dropped (the conversation builder adds the image token), and
* LISA's ``[SEG]`` becomes the paper's :data:`cora.constants.SEG_TOKEN` (``<SSEG>``).
"""
from __future__ import annotations

import random
import re
from typing import List

from cora import constants as C

# LISA utils/utils.py L14-42 (Apache-2.0), without the leading "<image>\n".
SHORT_QUESTION_LIST: List[str] = [
    "Can you segment the {class_name} in this image?",
    "Please segment the {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST: List[str] = [
    "{sent} Please respond with segmentation mask.",
    "{sent} Please output segmentation mask.",
]

# LISA's ANSWER_LIST with "[SEG]" replaced by the paper's segmentation token.
ANSWER_LIST: List[str] = [
    f"It is {C.SEG_TOKEN}.",
    f"Sure, {C.SEG_TOKEN}.",
    f"Sure, it is {C.SEG_TOKEN}.",
    f"Sure, the segmentation result is {C.SEG_TOKEN}.",
    f"{C.SEG_TOKEN}.",
]

# ----------------------------------------------------------------------------------------------
# Class phrases
# ----------------------------------------------------------------------------------------------
# Plural / mass forms of the Cityscapes class names, for queries that cover every instance ("the cars").
_CITYSCAPES_PLURAL = {
    "road": "road", "sidewalk": "sidewalks", "building": "buildings", "wall": "walls", "fence": "fences",
    "pole": "poles", "traffic light": "traffic lights", "traffic sign": "traffic signs",
    "vegetation": "vegetation", "terrain": "terrain", "sky": "sky", "person": "people", "rider": "riders",
    "car": "cars", "truck": "trucks", "bus": "buses", "train": "trains", "motorcycle": "motorcycles",
    "bicycle": "bicycles",
}
# Classes named by a mass noun: "the vegetation" takes "is", "the cars" takes "are".
_MASS_CLASSES = ("road", "vegetation", "terrain", "sky")

_ARTICLE = re.compile(r"^(the|a|an)(\s+|$)", re.IGNORECASE)


def _check_class(dataset: str, class_id: int) -> None:
    names = C.class_names(dataset)
    if not 0 <= int(class_id) < len(names):
        raise ValueError(f"class id {class_id} is outside 0..{len(names) - 1} for {dataset}")


def class_phrases(dataset: str, class_id: int) -> List[str]:
    """Names usable in queries for a class: the class name."""
    _check_class(dataset, class_id)
    return [C.CITYSCAPES_CLASSES[int(class_id)]]


def singular_phrases(dataset: str, class_id: int) -> List[str]:
    """Singular nouns that denote one instance of a class: "car"."""
    _check_class(dataset, class_id)
    return [C.CITYSCAPES_CLASSES[int(class_id)]]


def singular_phrase(dataset: str, class_id: int) -> str:
    """Primary singular noun of a class (used to name an anchor), e.g. "car"."""
    return singular_phrases(dataset, class_id)[0]


def plural_phrases(dataset: str, class_id: int) -> List[str]:
    """Plural (or mass) phrases that denote every instance of a class: "cars"."""
    _check_class(dataset, class_id)
    return [_CITYSCAPES_PLURAL[C.CITYSCAPES_CLASSES[int(class_id)]]]


def is_mass_phrase(dataset: str, class_id: int) -> bool:
    """True when the plural phrase of a class is a mass noun that takes "is" ("the vegetation is ...")."""
    _check_class(dataset, class_id)
    return dataset == "cityscapes" and C.CITYSCAPES_CLASSES[int(class_id)] in _MASS_CLASSES


# ----------------------------------------------------------------------------------------------
# Query builders
# ----------------------------------------------------------------------------------------------
def _check_text(text: str, what: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError(f"{what} is empty")
    for token in (C.IMAGE_TOKEN, C.SEG_TOKEN, C.REF_TOKEN, C.REF_MASK_TOKEN, C.REF_BOX_TOKEN):
        if token in text:
            raise ValueError(f"{what} must not contain the special token {token}: {text!r}")
    return text


def semantic_query(class_phrase: str, rng: random.Random) -> str:
    """One of LISA's short templates filled with a lower-cased class phrase (Sec. 3.4.2)."""
    phrase = _check_text(class_phrase, "class phrase").lower()
    return rng.choice(SHORT_QUESTION_LIST).format(class_name=phrase)


def long_query(sentence: str, rng: random.Random) -> str:
    """One of LISA's long templates wrapped around a free-form sentence."""
    return rng.choice(LONG_QUESTION_LIST).format(sent=_check_text(sentence, "sentence"))


def random_answer(rng: random.Random) -> str:
    """One of LISA's answers; it always contains exactly one segmentation token."""
    return rng.choice(ANSWER_LIST)


def wrap_description(description: str, rng: random.Random) -> str:
    """Turn an attribute description of a target into a segmentation query (template fallback).

    The description is a noun phrase such as "four wheeled vehicle used for personal transport"; a leading
    article and a final full stop are dropped, then one of LISA's short templates is filled. The
    LLM-written rewrites of the attribute set (supplement Table 4 + Sec. 3.4.2) do not go through here.
    """
    text = _check_text(description, "description").rstrip(" .!?;:")
    text = _ARTICLE.sub("", text)
    if not text:
        raise ValueError(f"description {description!r} holds no noun phrase")
    if len(text) > 1 and text[1].islower():
        text = text[0].lower() + text[1:]
    return rng.choice(SHORT_QUESTION_LIST).format(class_name=text)
