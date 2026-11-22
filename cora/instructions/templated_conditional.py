"""Rule-based, class-level conditional instructions (Fig. 2, Sec. 3.1).

The paper's own illustrations of the conditional set name the target by its class and cover every instance
that satisfies the relation: "Segment the cars on the left of <REF>" (Fig. 2). This module builds that set
without an LLM from 2D geometry, from the connected-component regions of the label maps (supplement Sec. 4).
It is the ``templated`` variant of the conditional set (``Config.instructions_path(split, "conditional", "templated")``).

Construction of one record:

* anchor: one region at least ``MIN_ANCHOR_FRAC`` of the image area (1e-3: 2,098 px of a 2048x1024
  Cityscapes frame; the threshold is rounded up), or, with
  ``use_anchor_groups`` (default on for Cityscapes, where Fig. 2 shows "two pedestrians" and "two motorcycles" as
  anchor thumbnails), a pair of neighbouring regions of one instance class
  (:func:`cora.data.relations.anchor_groups`) whose summed area reaches that threshold and whose members are
  each at least ``MIN_TARGET_FRAC`` of the image; the pair is rendered as one ``<REF>`` mask;
* relation: one of :func:`cora.data.relations.relations_for` (the 2D relations; plus "in front of" and
  "behind" on Cityscapes), evaluated in the image (camera) frame with the thresholds documented in
  :mod:`cora.data.relations`;
* target: the UNION of all regions of one class that satisfy the relation with the anchor (at least one),
  counting only regions of at least ``MIN_TARGET_FRAC`` of the image area (2e-4: 420 px of a Cityscapes
  frame); smaller fragments are neither in the target nor counted. The anchor is
  never among the targets (for a group anchor none of its members is), but a target class may equal the anchor class;
* at most ``cfg.rule_sets.templated_conditional_per_image`` records per image, drawn so that different
  anchors come first (one record per anchor in each round), with a random number generator seeded from
  ``cfg.rule_sets.seed`` and the image id, so the output does not depend on the order of the images.

Query wording: one of ``QUESTION_TEMPLATES`` with a target phrase, a relation phrase and the anchor
placeholder ``<REF>``. The target phrase is plural (or a mass noun) when the target holds several regions;
when exactly one region of a countable class qualifies it is singular, as in Fig. 2 ("Segment the bicycle on
the right of <REF>"), so the grammatical number matches the ground truth. With ``anchor_name="include"`` the anchor class precedes
the placeholder ("... above the car <REF>"); the default ``"omit"`` follows Fig. 2 and supp. Fig. 1.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from cora import constants as C
from cora.config import Config
from cora.data import templates
from cora.data.adapters import DatasetAdapter
from cora.data.objects import ImageObjects, ensure_objects
from cora.data.relations import (
    ALL_RELATIONS, RELATIONS, Region, anchor_groups, region_members, related_by_class, relations_for, resolve_thresholds,
)
from cora.data.schema import InstructionRecord, ObjectRef

MIN_ANCHOR_FRAC = 1e-3
MIN_TARGET_FRAC = 2e-4
ANCHOR_NAME_MODES = ("omit", "include")

# Phrasings of each relation inside a query. "above" / "surrounding" etc. keep the paper's words.
RELATION_PHRASES: Dict[str, Tuple[str, ...]] = {
    "above": ("above",),
    "below": ("below", "beneath"),
    "to the left of": ("on the left of", "to the left of"),
    "to the right of": ("on the right of", "to the right of"),
    "near": ("near", "close to"),
    "around": ("around",),
    "surrounding": ("surrounding",),
    "in front of": ("in front of",),
    "behind": ("behind", "at the back of"),
}
QUESTION_TEMPLATES: Tuple[str, ...] = (
    "Segment the {targets} {relation} {anchor}.",
    "Can you segment the {targets} {relation} {anchor}?",
    "Please segment the {targets} {relation} {anchor}.",
    "Segment all the {targets} that {be} {relation} {anchor}.",
    "Find and segment the {targets} {relation} {anchor}.",
)


@dataclass(frozen=True)
class Candidate:
    """An anchor, a relation and the regions of one class that satisfy it."""

    anchor: Region
    label_id: int
    relation: str
    targets: Tuple[Region, ...]


def area_threshold(objects: ImageObjects, fraction: float, floor: int = 1) -> int:
    """Pixels equal to ``fraction`` of the image area (rounded up, at least ``floor``)."""
    return max(int(floor), int(math.ceil(fraction * objects.height * objects.width)))


def anchor_candidates(
    objects: ImageObjects,
    anchor: Region,
    relations: Sequence[str],
    min_target_area: int,
    **thresholds: float,
) -> List[Candidate]:
    """Every (class, relation) pair with at least one qualifying target region for one anchor."""
    found: List[Candidate] = []
    for relation in relations:
        grouped = related_by_class(objects, anchor, relation, min_area=min_target_area, **thresholds)
        for label_id in sorted(grouped):
            found.append(Candidate(anchor, label_id, relation, tuple(grouped[label_id])))
    return found


def select_candidates(
    objects: ImageObjects,
    rng: random.Random,
    count: int,
    *,
    relations: Sequence[str] = RELATIONS,
    min_anchor_area: int,
    min_target_area: int,
    accept: Optional[Callable[[Candidate], bool]] = None,
    use_anchor_groups: bool = False,
    **thresholds: float,
) -> List[Candidate]:
    """Draw up to ``count`` candidates; every round takes one random candidate from each anchor in turn.

    Anchors are visited in a random order and their candidate lists are computed only when needed.
    ``accept`` filters candidates (for example to those with a single target region). With
    ``use_anchor_groups`` the pairs of :func:`cora.data.relations.anchor_groups` (members of at least
    ``min_target_area`` pixels, together at least ``min_anchor_area``) are anchors next to the single regions.
    """
    anchors: List[Region] = [r for r in objects.regions if r.area >= min_anchor_area]
    if use_anchor_groups:
        anchors += [g for g in anchor_groups(objects, min_target_area, **thresholds) if g.area >= min_anchor_area]
    rng.shuffle(anchors)
    pools: Dict[str, List[Candidate]] = {}
    chosen: List[Candidate] = []
    while len(chosen) < count:
        progressed = False
        for anchor in anchors:
            if len(chosen) >= count:
                break
            if anchor.region_id not in pools:
                pool = anchor_candidates(objects, anchor, relations, min_target_area, **thresholds)
                pools[anchor.region_id] = [c for c in pool if accept is None or accept(c)]
            pool = pools[anchor.region_id]
            if pool:
                chosen.append(pool.pop(rng.randrange(len(pool))))
                progressed = True
        if not progressed:
            break
    return chosen


def render_query(dataset: str, candidate: Candidate, rng: random.Random, anchor_name: str = "omit",
                 target_phrase: Optional[str] = None) -> Tuple[str, int]:
    """Query text and template index for a candidate.

    The target is named by a plural (or mass) phrase of its class, by a singular phrase when exactly one
    region of a countable class qualifies, or by
    ``target_phrase`` when given (the offline generator passes a plural attribute phrase).
    """
    if anchor_name not in ANCHOR_NAME_MODES:
        raise ValueError(f"anchor_name must be one of {ANCHOR_NAME_MODES}, got {anchor_name!r}")
    index = rng.randrange(len(QUESTION_TEMPLATES))
    single = False
    if target_phrase is None:
        mass = templates.is_mass_phrase(dataset, candidate.label_id)
        single = len(candidate.targets) == 1 and not mass
        if single:
            phrase = rng.choice(templates.singular_phrases(dataset, candidate.label_id))
        else:
            phrase = rng.choice(templates.plural_phrases(dataset, candidate.label_id))
        be = "are" if not (mass or single) else "is"
    else:
        phrase, be = target_phrase, "are"
    anchor = C.REF_TOKEN
    if anchor_name == "include":
        grouped = len(region_members(candidate.anchor)) > 1
        noun = (templates.plural_phrases if grouped else templates.singular_phrases)(dataset, candidate.anchor.label_id)[0]
        anchor = f"the {noun} {C.REF_TOKEN}"
    relation = rng.choice(RELATION_PHRASES[candidate.relation])
    template = QUESTION_TEMPLATES[index]
    if single:
        template = template.replace("all the ", "the ")  # "Segment all the car" is not English
    return template.format(targets=phrase, relation=relation, anchor=anchor, be=be), index


def conditional_record(
    dataset: str, split: str, image_id: str, candidate: Candidate, query: str, meta: Dict[str, object],
    record_id: str,
) -> InstructionRecord:
    """Validated conditional :class:`InstructionRecord` of a candidate."""
    names = C.class_names(dataset)
    record = InstructionRecord(
        id=record_id,
        dataset=dataset,
        split=split,
        image_id=image_id,
        set=C.SET_CONDITIONAL,
        query=query,
        target=ObjectRef(candidate.label_id, names[candidate.label_id], [r.region_id for r in candidate.targets]),
        anchor=ObjectRef(candidate.anchor.label_id, names[candidate.anchor.label_id],
                         [m.region_id for m in region_members(candidate.anchor)]),
        relation=candidate.relation,
        meta=dict(meta),
    )
    record.validate()
    return record


def build_templated_conditional(
    cfg: Config,
    adapter: DatasetAdapter,
    split: str,
    image_ids: Sequence[str],
    *,
    anchor_name: str = "omit",
    relations: Optional[Sequence[str]] = None,
    use_anchor_groups: Optional[bool] = None,
    per_image: Optional[int] = None,
    min_anchor_frac: float = MIN_ANCHOR_FRAC,
    min_target_frac: float = MIN_TARGET_FRAC,
    **thresholds: float,
) -> Iterator[InstructionRecord]:
    """Class-level conditional records of the images (Fig. 2).

    ``per_image`` defaults to ``cfg.rule_sets.templated_conditional_per_image``; ``relations`` to
    :func:`cora.data.relations.relations_for` of the dataset; ``use_anchor_groups`` to True on Cityscapes and False
    on other datasets; ``thresholds`` are the relation thresholds of :mod:`cora.data.relations`. Images without a
    qualifying anchor / target pair yield nothing.
    """
    if relations is None:
        relations = relations_for(adapter.name)
    if use_anchor_groups is None:
        use_anchor_groups = adapter.name == "cityscapes"
    unknown = [r for r in relations if r not in ALL_RELATIONS]
    if unknown:
        raise ValueError(f"unknown relation(s) {unknown}; expected a subset of {ALL_RELATIONS}")
    if anchor_name not in ANCHOR_NAME_MODES:
        raise ValueError(f"anchor_name must be one of {ANCHOR_NAME_MODES}, got {anchor_name!r}")
    resolve_thresholds(thresholds)
    count = cfg.rule_sets.templated_conditional_per_image if per_image is None else per_image
    if count < 0:
        raise ValueError(f"per_image must be >= 0, got {count}")
    for image_id in image_ids:
        if count == 0:
            return
        objects = ensure_objects(cfg, adapter, split, image_id)
        rng = random.Random(f"{cfg.rule_sets.seed}|templated|{split}|{image_id}")
        picked = select_candidates(
            objects, rng, count, relations=relations,
            min_anchor_area=area_threshold(objects, min_anchor_frac, cfg.data.min_region_area),
            min_target_area=area_threshold(objects, min_target_frac, cfg.data.min_region_area),
            use_anchor_groups=use_anchor_groups, **thresholds,
        )
        for candidate in picked:
            query, template_index = render_query(adapter.name, candidate, rng, anchor_name)
            meta = {"generator": "rule", "style": "templated", "template": template_index, "anchor_name": anchor_name,
                    "num_targets": len(candidate.targets), "num_anchor_regions": len(region_members(candidate.anchor)),
                    "seed": cfg.rule_sets.seed}
            record_id = f"{image_id}:templated:{candidate.label_id}:{candidate.anchor.region_id}:{candidate.relation}"
            yield conditional_record(adapter.name, split, image_id, candidate, query, meta, record_id)
