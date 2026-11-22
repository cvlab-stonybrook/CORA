"""Rule-based stand-in for the LLM-generated attribute and conditional sets (testing only).

The real attribute set (supplement Table 4) and conditional set (supplement Table 5) are LLM-written and come with
the provided instruction files. This module exists ONLY so that the training pipeline can be exercised end to end
without them: geometry comes from :mod:`cora.data.relations`, descriptions are fixed attribute phrases per class
(``ATTRIBUTE_PHRASES``). Differences from the real sets, by construction:

* attribute records describe the whole class of the image (target = every region of the class above the
  minimum area), because a fixed phrase cannot tell instances of one class apart;
* conditional records name the target by an attribute phrase of its class instead of the class name and use
  the class-level target of :mod:`cora.instructions.templated_conditional` (union of all qualifying regions).

Every record carries ``meta["generator"] = "offline"``. :func:`refuse_if_llm_generated` implements the
refusal rule: stand-in data is never written into a data folder that already holds
LLM-generated files of the same set (records whose ``meta["generator"]`` is neither ``"offline"`` nor
``"rule"``), so it cannot be mixed into real instruction sets.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

from cora import constants as C
from cora.config import Config
from cora.data import templates
from cora.data.adapters import DatasetAdapter
from cora.data.objects import ensure_objects
from cora.data.relations import RELATIONS
from cora.data.schema import InstructionRecord, ObjectRef
from cora.instructions.templated_conditional import (
    MIN_ANCHOR_FRAC, MIN_TARGET_FRAC, area_threshold, conditional_record, render_query, select_candidates,
)

OFFLINE_SETS = (C.SET_ATTRIBUTE, C.SET_CONDITIONAL)
GENERATOR = "offline"
NON_LLM_GENERATORS = (GENERATOR, "rule")

# Plural / mass noun phrases that describe the whole class by appearance and function (never the class name).
ATTRIBUTE_PHRASES: Dict[str, Dict[int, List[str]]] = {
    "cityscapes": {
        0: ["paved asphalt driving surfaces", "smooth tarmac lanes used by motor traffic", "flat dark ground marked with lane lines"],
        1: ["raised paved walkways for people on foot", "curbed footpaths made of paving slabs", "paved strips reserved for walkers"],
        2: ["roofed structures with windows and doors", "houses, shops and office blocks", "tall man-made structures with facades"],
        3: ["solid free-standing masonry barriers", "windowless brick or concrete partitions", "stone enclosures that bound properties"],
        4: ["see-through barriers of wire, slats or bars", "wooden or metal boundary railings", "light pickets and mesh enclosures"],
        5: ["thin vertical posts", "slender upright shafts that carry lamps and cables", "tall narrow metal or wooden columns"],
        6: ["lamp units with red, amber and green lights", "signal heads that regulate crossings", "colour-changing signal boxes for motorists"],
        7: ["flat plates carrying road symbols", "reflective boards that tell drivers the rules", "warning and regulatory panels with pictograms"],
        8: ["leafy trees and bushes", "green foliage on woody stems", "hedges and tree crowns"],
        9: ["grass and bare soil", "natural lawns and dirt ground", "unpaved earthy ground cover"],
        10: ["open air with clouds", "bright expanse of clouds and open air", "blue or grey atmosphere"],
        11: ["human beings walking on foot", "pedestrians", "walkers carrying bags and umbrellas"],
        12: ["humans sitting on bikes and scooters", "cyclists in helmets", "bikers holding handlebars"],
        13: ["small private passenger automobiles", "sedans and hatchbacks used for personal transport", "family motor vehicles with a closed cabin"],
        14: ["heavy freight haulers with cargo boxes", "delivery lorries with a separate load area", "large goods vehicles with a driver's cab"],
        15: ["long public transit vehicles with many windows", "coaches that carry numerous passengers", "tall boxy vehicles with route displays"],
        16: ["rail-bound vehicles on steel tracks", "trams and railway carriages", "electric transit cars running on rails"],
        17: ["motorised bikes with handlebars and a saddle", "mopeds and scooters with a motor", "petrol-driven cycles with an exhaust pipe"],
        18: ["pedal-powered cycles with spoked wheels", "human-powered bikes with a chain", "slim-tyred frames with handlebars and a saddle"],
    },
}


class OfflineRefusal(RuntimeError):
    """Writing stand-in data would mix it with LLM-generated instruction files."""


def _generators_in(path: Path) -> set:
    found = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                found.add(json.loads(line).get("meta", {}).get("generator"))
            except (ValueError, AttributeError) as err:
                raise ValueError(f"{path}:{line_no}: not a valid instruction record ({err})") from None
    return found


def refuse_if_llm_generated(cfg: Config, split: str, set_name: str) -> None:
    """Raise :class:`OfflineRefusal` when a file of ``set_name`` holds records not made by rules or by this module.

    Checks the set file and its variants (``Config.instruction_files``). Files that contain only offline
    records are fine: they are replaced by a new offline run.
    """
    for path in cfg.instruction_files(split, set_name):
        foreign = sorted(str(g) for g in _generators_in(path) if g not in NON_LLM_GENERATORS)
        if foreign:
            raise OfflineRefusal(
                f"{path} already holds records from generator(s) {foreign}; the offline stand-in never writes next to "
                "LLM-generated instruction sets. Point paths.data_dir at a separate folder to test with stand-in data."
            )


def build_offline_records(
    cfg: Config,
    adapter: DatasetAdapter,
    split: str,
    image_ids: Sequence[str],
    sets: Sequence[str],
    *,
    anchor_name: str = "omit",
    relations: Sequence[str] = RELATIONS,
    **thresholds: float,
) -> Iterator[InstructionRecord]:
    """Stand-in records of the ``attribute`` and / or ``conditional`` set for every image.

    Attribute: up to ``cfg.rule_sets.attribute_targets_per_image`` classes of the image, each with up to
    ``cfg.rule_sets.qa_per_target`` distinct phrases. Conditional: up to
    ``cfg.rule_sets.templated_conditional_per_image`` records per image.
    """
    wanted = list(dict.fromkeys(sets))
    bad = [s for s in wanted if s not in OFFLINE_SETS]
    if bad or not wanted:
        raise ValueError(f"sets must be a non-empty subset of {OFFLINE_SETS} (the semantic set is rule-based already), got {list(sets)}")
    phrases = ATTRIBUTE_PHRASES.get(adapter.name)
    if phrases is None:
        raise ValueError(f"no attribute phrases for dataset {adapter.name!r}")
    names = adapter.class_names
    for image_id in image_ids:
        objects = ensure_objects(cfg, adapter, split, image_id)
        min_target = area_threshold(objects, MIN_TARGET_FRAC, cfg.data.min_region_area)
        if C.SET_ATTRIBUTE in wanted:
            rng = random.Random(f"{cfg.rule_sets.seed}|offline-attribute|{split}|{image_id}")
            present = sorted({r.label_id for r in objects.regions if r.area >= min_target})
            rng.shuffle(present)
            for class_id in present[: cfg.rule_sets.attribute_targets_per_image]:
                region_ids = [r.region_id for r in objects.of_class(class_id) if r.area >= min_target]
                options = phrases[class_id]
                for k, description in enumerate(rng.sample(options, min(cfg.rule_sets.qa_per_target, len(options)))):
                    record = InstructionRecord(
                        id=f"{image_id}:offline:attribute:{class_id}:{k}",
                        dataset=adapter.name, split=split, image_id=image_id, set=C.SET_ATTRIBUTE,
                        query=templates.wrap_description(description, rng),
                        target=ObjectRef(class_id, names[class_id], region_ids),
                        description=description,
                        meta={"generator": GENERATOR, "target_scope": "class", "seed": cfg.rule_sets.seed},
                    )
                    record.validate()
                    yield record
        if C.SET_CONDITIONAL in wanted:
            rng = random.Random(f"{cfg.rule_sets.seed}|offline-conditional|{split}|{image_id}")
            picked = select_candidates(
                objects, rng, cfg.rule_sets.templated_conditional_per_image, relations=relations,
                min_anchor_area=area_threshold(objects, MIN_ANCHOR_FRAC, cfg.data.min_region_area),
                min_target_area=min_target, **thresholds,
            )
            for candidate in picked:
                description = rng.choice(phrases[candidate.label_id])
                query, template_index = render_query(adapter.name, candidate, rng, anchor_name, target_phrase=description)
                record = conditional_record(
                    adapter.name, split, image_id, candidate, query,
                    {"generator": GENERATOR, "style": "offline", "template": template_index, "anchor_name": anchor_name,
                     "num_targets": len(candidate.targets), "seed": cfg.rule_sets.seed},
                    f"{image_id}:offline:conditional:{candidate.label_id}:{candidate.anchor.region_id}:{candidate.relation}",
                )
                record.description = description
                yield record
