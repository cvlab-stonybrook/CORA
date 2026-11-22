"""Semantic segmentation instruction set (paper Sec. 3.4.2).

"We construct these instructions by fitting class labels into segmentation query templates." One class-level
record is written per class present in an image. The stored query is one template instance
(:func:`cora.data.templates.semantic_query`); the dataset draws a fresh template at load time, which is why
``meta["class_level"]`` is true. The target is the whole class (empty ``region_ids``).
"""
from __future__ import annotations

import random
from typing import Iterator, Sequence

import numpy as np

from cora import constants as C
from cora.config import Config
from cora.data import templates
from cora.data.adapters import DatasetAdapter
from cora.data.schema import InstructionRecord, ObjectRef


def build_semantic_records(
    cfg: Config,
    adapter: DatasetAdapter,
    split: str,
    image_ids: Sequence[str],
    *,
    min_pixels: int = 1,
) -> Iterator[InstructionRecord]:
    """One class-level record per class with at least ``min_pixels`` pixels in each image's label map."""
    if min_pixels < 1:
        raise ValueError(f"min_pixels must be at least 1, got {min_pixels}")
    names = adapter.class_names
    for image_id in image_ids:
        label = adapter.load_label(image_id)
        classes, counts = np.unique(label, return_counts=True)
        rng = random.Random(f"{cfg.rule_sets.seed}|semantic|{split}|{image_id}")
        for class_id, count in zip(classes.tolist(), counts.tolist()):
            if class_id == C.IGNORE_LABEL or count < min_pixels:
                continue
            if not 0 <= class_id < len(names):
                raise ValueError(f"{image_id}: label map holds class id {class_id}, outside the {len(names)} classes of {adapter.name}")
            phrase = rng.choice(templates.class_phrases(adapter.name, class_id))
            record = InstructionRecord(
                id=f"{image_id}:semantic:{class_id}",
                dataset=adapter.name,
                split=split,
                image_id=image_id,
                set=C.SET_SEMANTIC,
                query=templates.semantic_query(phrase, rng),
                target=ObjectRef(class_id, names[class_id], []),
                meta={"generator": "rule", "class_level": True, "pixels": int(count), "seed": cfg.rule_sets.seed},
            )
            record.validate()
            yield record
