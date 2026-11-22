"""Deterministic 2D spatial relations between regions (Fig. 2).

The class-level conditional set (Fig. 2, "Segment the cars on the left of <REF>") is built from these rules.
The paper gives no thresholds, so every number below is an implementation choice and can be overridden per
call (``**thresholds``).

Frame of reference: the image (camera) frame, as in the LLM prompt of supplement Table 5. x grows to the
right and y downwards, so "to the left of" means smaller x on screen.

Region quantities come from :class:`cora.data.objects.Region` (bounding box ``x0, y0, x1, y1`` with x1 / y1
exclusive, centroid ``cx, cy``); ``dx = target.cx - anchor.cx`` and ``dy = target.cy - anchor.cy``.

Directions (all four conditions must hold; a target therefore satisfies at most one direction):

* "to the left of":  ``dx <= -m_x``, ``target.x1 <= anchor.cx`` and ``|dx| >= |dy|``
* "to the right of": ``dx >= +m_x``, ``target.x0 >= anchor.cx`` and ``|dx| >= |dy|``
* "above":           ``dy <= -m_y``, ``target.y1 <= anchor.cy`` and ``|dy| >  |dx|``
* "below":           ``dy >= +m_y``, ``target.y0 >= anchor.cy`` and ``|dy| >  |dx|``

with the margins ``m_x = direction_margin_frac * width`` and ``m_y = direction_margin_frac * height``
(default 0.02). The box condition keeps the target from reaching past the anchor's centre; the dominance
condition keeps it inside the 90 degree sector of its direction.

Distances use the gap = the smallest Euclidean distance between a pixel of the target and a pixel of the
anchor (1 for touching regions), measured with the region masks when they are available and with the
bounding boxes otherwise (the box gap never exceeds the mask gap). ``diag`` is the image diagonal.

* "near":        ``gap <= near_frac * diag``                      (default 0.05)
* "around":      ``gap <= around_frac * diag`` (default 0.10) for every member, and the members of the class
                 together must lie in at least ``around_min_octants`` (default 3) of the 8 45-degree sectors
                 of the compass seen from the anchor centroid (sector of each member's centroid)
* "surrounding": ``gap <= near_frac * diag`` for every member, and the members of the class together must
                 cover at least ``surround_cover`` (default 0.5) of the anchor's ring, the pixels of the
                 image outside the anchor and within ``near_frac * diag`` of it

Depth relations (Cityscapes only; the camera frame of supplement Table 5, "the closer object is 'in front of'
the farther"):

* "in front of": ``target.y1 - anchor.y1 >= m_d`` and the horizontal extents overlap
* "behind":      ``anchor.y1 - target.y1 >= m_d`` and the horizontal extents overlap

For an anchor made of two regions the depth inequality must hold against the bottom edge of each member (the two
stand at different depths, and "behind <REF>" must be true for both); the horizontal overlap is measured against
the box around the pair.

Both use the ground-plane cue: an object standing on the road is nearer to the camera the lower its bottom
edge ``y1`` lies in the image. ``m_d = depth_margin_frac * height`` (default 0.03); the overlap is the shared
width divided by the narrower box width and must reach ``depth_overlap_frac`` (default 0.25), so that the two
objects are on the same line of sight rather than side by side. The cue only holds for objects that stand on
the ground, so both regions must belong to a class of :data:`GROUND_CLASS_NAMES` (person, car, truck, bus,
train, motorcycle, bicycle; a rider sits on a vehicle, so its bottom edge is not where it meets the ground and
"the rider behind the bicycle" would describe the rider of that bicycle).

Anchors can be groups: Fig. 2 of the paper shows an anchor thumbnail with two pedestrians or two motorcycles.
:func:`anchor_groups` pairs every instance region with its nearest same-class neighbour closer than
``group_frac * diag`` (default 0.03); the pair is a :class:`RegionGroup` that all functions below accept in
place of a :class:`Region`. A group has the union box, the area-weighted centroid and the union mask of its
members; none of its members is ever a target.

"around" and "surrounding" describe a group of regions of ONE class: :func:`related_regions` returns either
every member of the class's group or nothing (the group condition is checked per class, never across
classes). :func:`relation_holds` is the per-region necessary condition (the distance rule); the group
conditions need the whole image. ``min_area`` (default 1) drops target regions that are smaller than that
many pixels before any rule is applied. The anchor itself is never a target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from cora.data.objects import ImageObjects, Region

RELATIONS: Tuple[str, ...] = (
    "above", "below", "to the left of", "to the right of", "near", "around", "surrounding",
)  # the 2D relations, valid for every dataset
DEPTH_RELATIONS: Tuple[str, ...] = ("in front of", "behind")  # Cityscapes only (ground-plane cue)
ALL_RELATIONS: Tuple[str, ...] = RELATIONS + DEPTH_RELATIONS
INSTANCE_CLASS_NAMES: FrozenSet[str] = frozenset(
    {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
)  # Cityscapes classes of countable objects
GROUND_CLASS_NAMES: FrozenSet[str] = INSTANCE_CLASS_NAMES - {"rider"}  # those whose bottom edge is on the ground
DIRECTIONS = ("above", "below", "to the left of", "to the right of")
DISTANCE_RELATIONS = ("near", "around", "surrounding")
GROUP_RELATIONS = ("around", "surrounding")

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "direction_margin_frac": 0.02,
    "near_frac": 0.05,
    "around_frac": 0.10,
    "around_min_octants": 3,
    "surround_cover": 0.5,
    "min_area": 1,
    "depth_margin_frac": 0.03,
    "depth_overlap_frac": 0.25,
    "group_frac": 0.03,
}


def resolve_thresholds(thresholds: Dict[str, float]) -> Dict[str, float]:
    """Defaults updated with the given values; unknown names and out-of-range values raise ``ValueError``."""
    unknown = sorted(set(thresholds) - set(DEFAULT_THRESHOLDS))
    if unknown:
        raise ValueError(f"unknown relation threshold(s) {unknown}; valid: {sorted(DEFAULT_THRESHOLDS)}")
    merged = {**DEFAULT_THRESHOLDS, **thresholds}
    for name in ("direction_margin_frac", "near_frac", "around_frac", "depth_margin_frac", "group_frac"):
        if not 0 <= merged[name] <= 1:
            raise ValueError(f"{name} must lie in [0, 1], got {merged[name]}")
    if not 0 < merged["surround_cover"] <= 1:
        raise ValueError(f"surround_cover must lie in (0, 1], got {merged['surround_cover']}")
    if not 0 < merged["depth_overlap_frac"] <= 1:
        raise ValueError(f"depth_overlap_frac must lie in (0, 1], got {merged['depth_overlap_frac']}")
    if not 1 <= merged["around_min_octants"] <= 8:
        raise ValueError(f"around_min_octants must lie in 1..8, got {merged['around_min_octants']}")
    if merged["min_area"] < 1:
        raise ValueError(f"min_area must be at least 1, got {merged['min_area']}")
    return merged


@dataclass
class RegionGroup(Region):
    """Several regions of one class used together as one anchor (Fig. 2: "two pedestrians").

    The inherited fields describe the union: summed area, union box, area-weighted centroid. ``region_id``
    joins the member ids with "+", ``index`` is the first member's.
    """

    members: Tuple[Region, ...] = ()


def merge_regions(members: Sequence[Region]) -> RegionGroup:
    """The :class:`RegionGroup` of two or more regions of one class."""
    members = tuple(members)
    if len(members) < 2:
        raise ValueError("a region group needs at least two regions")
    if len({m.label_id for m in members}) != 1:
        raise ValueError(f"the regions of a group must share one class, got {sorted({m.label_name for m in members})}")
    area = sum(m.area for m in members)
    centroid = (sum(m.centroid[0] * m.area for m in members) / area, sum(m.centroid[1] * m.area for m in members) / area)
    bbox = (min(m.bbox[0] for m in members), min(m.bbox[1] for m in members),
            max(m.bbox[2] for m in members), max(m.bbox[3] for m in members))
    first = members[0]
    return RegionGroup(index=first.index, region_id="+".join(m.region_id for m in members), label_id=first.label_id,
                       label_name=first.label_name, area=area, bbox=bbox, centroid=centroid, members=members)


def region_members(region: Region) -> Tuple[Region, ...]:
    """The regions behind ``region``: its members for a group, else the region itself."""
    return region.members if isinstance(region, RegionGroup) else (region,)


def relations_for(dataset: str) -> Tuple[str, ...]:
    """Relations the rule-based conditional set uses on a dataset: 2D relations, plus depth for Cityscapes."""
    return ALL_RELATIONS if dataset == "cityscapes" else RELATIONS


def _check_relation(relation: str) -> None:
    if relation not in ALL_RELATIONS:
        raise ValueError(f"unknown relation {relation!r}; expected one of {ALL_RELATIONS}")


def _depth_holds(target: Region, anchor: Region, relation: str, image_hw: Tuple[int, int], th: Dict[str, float]) -> bool:
    """Ground-plane depth rule of "in front of" / "behind" (see module docs)."""
    if target.label_name not in GROUND_CLASS_NAMES or anchor.label_name not in GROUND_CLASS_NAMES:
        return False
    margin = th["depth_margin_frac"] * image_hw[0]
    # A group anchor is two objects at different depths: the relation must hold against each of them, or the query
    # ("behind <REF>") would be false for one member.
    bottoms = [member.bbox[3] for member in region_members(anchor)]
    if relation == "in front of":
        if any(target.bbox[3] - bottom < margin for bottom in bottoms):
            return False
    elif any(bottom - target.bbox[3] < margin for bottom in bottoms):
        return False
    overlap = min(target.bbox[2], anchor.bbox[2]) - max(target.bbox[0], anchor.bbox[0])
    narrower = min(target.bbox[2] - target.bbox[0], anchor.bbox[2] - anchor.bbox[0])
    return overlap > 0 and overlap >= th["depth_overlap_frac"] * narrower


def _direction_holds(target: Region, anchor: Region, relation: str, image_hw: Tuple[int, int], margin_frac: float) -> bool:
    height, width = image_hw
    tx, ty = target.centroid
    ax, ay = anchor.centroid
    dx, dy = tx - ax, ty - ay
    if relation == "to the left of":
        return dx <= -margin_frac * width and target.bbox[2] <= ax and abs(dx) >= abs(dy)
    if relation == "to the right of":
        return dx >= margin_frac * width and target.bbox[0] >= ax and abs(dx) >= abs(dy)
    if relation == "above":
        return dy <= -margin_frac * height and target.bbox[3] <= ay and abs(dy) > abs(dx)
    return dy >= margin_frac * height and target.bbox[1] >= ay and abs(dy) > abs(dx)  # below


def box_gap(a: Region, b: Region) -> float:
    """Smallest distance between a pixel of ``a``'s box and a pixel of ``b``'s box (1 when they touch).

    For a :class:`RegionGroup` the boxes of its members are used, not the union box.
    """
    if isinstance(a, RegionGroup) or isinstance(b, RegionGroup):
        return min(box_gap(x, y) for x in region_members(a) for y in region_members(b))
    gap_x = max(0, a.bbox[0] - (b.bbox[2] - 1), b.bbox[0] - (a.bbox[2] - 1))
    gap_y = max(0, a.bbox[1] - (b.bbox[3] - 1), b.bbox[1] - (a.bbox[3] - 1))
    return max(1.0, math.hypot(gap_x, gap_y))


def mask_gap(target_mask: np.ndarray, anchor_mask: np.ndarray) -> float:
    """Smallest distance between a pixel of ``target_mask`` and a pixel of ``anchor_mask``."""
    if not target_mask.any() or not anchor_mask.any():
        raise ValueError("both masks must be non-empty to measure a gap")
    return float(ndimage.distance_transform_edt(~anchor_mask)[target_mask].min())


def _diagonal(image_hw: Tuple[int, int]) -> float:
    return math.hypot(*image_hw)


def _distance_ok(relation: str, gap: float, diag: float, th: Dict[str, float]) -> bool:
    radius = th["around_frac"] if relation == "around" else th["near_frac"]
    return gap <= radius * diag


def relation_holds(
    target: Region,
    anchor: Region,
    relation: str,
    image_hw: Tuple[int, int],
    *,
    target_mask: Optional[np.ndarray] = None,
    anchor_mask: Optional[np.ndarray] = None,
    **thresholds: float,
) -> bool:
    """Whether ``target`` satisfies ``relation`` with respect to ``anchor`` (per-region rule, see module docs).

    For "around" and "surrounding" this is the distance condition every member of the group must meet; the
    group conditions are checked by :func:`related_regions`. Pass both masks ((H, W) bool) to measure the
    distance on pixels instead of boxes.
    """
    _check_relation(relation)
    th = resolve_thresholds(thresholds)
    if target.region_id in {m.region_id for m in region_members(anchor)} or target.area < th["min_area"]:
        return False
    if relation in DIRECTIONS:
        return _direction_holds(target, anchor, relation, image_hw, th["direction_margin_frac"])
    if relation in DEPTH_RELATIONS:
        return _depth_holds(target, anchor, relation, image_hw, th)
    if (target_mask is None) != (anchor_mask is None):
        raise ValueError("pass both target_mask and anchor_mask, or neither")
    gap = box_gap(target, anchor) if target_mask is None else mask_gap(target_mask, anchor_mask)
    return _distance_ok(relation, gap, _diagonal(image_hw), th)


def _octant(anchor: Region, region: Region) -> int:
    """Index 0..7 of the 45 degree compass sector of ``region``'s centroid seen from ``anchor``'s centroid."""
    angle = math.atan2(region.centroid[1] - anchor.centroid[1], region.centroid[0] - anchor.centroid[0])
    return int(((angle + math.pi) / (2 * math.pi)) * 8) % 8


def _window(anchor: Region, radius: float, height: int, width: int) -> Tuple[int, int, int, int]:
    pad = int(math.ceil(radius)) + 1
    x0, y0, x1, y1 = anchor.bbox
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def related_by_class(objects: ImageObjects, anchor: Region, relation: str, **thresholds: float) -> Dict[int, List[Region]]:
    """Regions that satisfy ``relation`` with ``anchor`` (a region or a group), grouped by class id (classes
    without any are absent).

    Distances are measured on the region masks. Within a class the regions are in index order. The group
    conditions of "around" and "surrounding" are evaluated separately for every class.
    """
    _check_relation(relation)
    th = resolve_thresholds(thresholds)
    image_hw = (objects.height, objects.width)
    anchor_ids = {m.region_id for m in region_members(anchor)}
    candidates = [r for r in objects.regions if r.region_id not in anchor_ids and r.area >= th["min_area"]]
    if relation in DIRECTIONS or relation in DEPTH_RELATIONS:
        grouped: Dict[int, List[Region]] = {}
        for region in candidates:
            holds = (_direction_holds(region, anchor, relation, image_hw, th["direction_margin_frac"])
                     if relation in DIRECTIONS else _depth_holds(region, anchor, relation, image_hw, th))
            if holds:
                grouped.setdefault(region.label_id, []).append(region)
        return grouped
    return _distance_members(objects, anchor, candidates, relation, th)


def _distance_members(objects: ImageObjects, anchor: Region, candidates: List[Region], relation: str,
                      th: Dict[str, float]) -> Dict[int, List[Region]]:
    """Per class, the candidates within the distance of ``relation``, subject to its group condition."""
    diag = _diagonal((objects.height, objects.width))
    radius = (th["around_frac"] if relation == "around" else th["near_frac"]) * diag
    x0, y0, x1, y1 = _window(anchor, radius, objects.height, objects.width)
    crop = objects.region_map[y0:y1, x0:x1]
    # Distance of every window pixel to the anchor. Pixels farther than `radius` are irrelevant, and the
    # window holds every pixel within `radius` of the anchor, so the cropped transform is exact there.
    distance = ndimage.distance_transform_edt(~np.isin(crop, [m.index for m in region_members(anchor)]))
    grouped: Dict[int, List[Region]] = {}
    for region in candidates:
        rx0, ry0, rx1, ry1 = region.bbox
        if rx1 <= x0 or rx0 >= x1 or ry1 <= y0 or ry0 >= y1:
            continue
        inside = crop == region.index
        if inside.any() and distance[inside].min() <= radius:
            grouped.setdefault(region.label_id, []).append(region)
    if relation == "near" or not grouped:
        return grouped
    if relation == "around":
        return {c: m for c, m in grouped.items() if len({_octant(anchor, r) for r in m}) >= th["around_min_octants"]}
    ring = (distance > 0) & (distance <= radius)  # surrounding: share of the anchor's ring that a class covers
    ring_size = int(ring.sum())
    if ring_size == 0:
        return {}
    return {c: m for c, m in grouped.items()
            if np.isin(crop[ring], [r.index for r in m]).sum() / ring_size >= th["surround_cover"]}


def related_regions(objects: ImageObjects, anchor: Region, label_id: int, relation: str, **thresholds: float) -> List[Region]:
    """Regions of class ``label_id`` that satisfy ``relation`` with ``anchor`` (never the anchor itself)."""
    return related_by_class(objects, anchor, relation, **thresholds).get(int(label_id), [])


def anchor_groups(objects: ImageObjects, min_member_area: int = 1, **thresholds: float) -> List[RegionGroup]:
    """Pairs of neighbouring regions of one instance class, usable as group anchors (Fig. 2).

    Every instance region of at least ``min_member_area`` pixels is paired with its nearest same-class
    neighbour (by :func:`box_gap`, ties to the lower region id) when that gap is at most
    ``group_frac * diag``; a pair found from both sides appears once. Deterministic, sorted by member ids.
    """
    th = resolve_thresholds(thresholds)
    limit = th["group_frac"] * _diagonal((objects.height, objects.width))
    pairs: Dict[Tuple[str, str], RegionGroup] = {}
    by_class: Dict[int, List[Region]] = {}
    for region in objects.regions:
        if region.label_name in INSTANCE_CLASS_NAMES and region.area >= min_member_area:
            by_class.setdefault(region.label_id, []).append(region)
    for members in by_class.values():
        for region in members:
            others = [o for o in members if o.region_id != region.region_id]
            if not others:
                continue
            nearest = min(others, key=lambda o: (box_gap(region, o), o.region_id))
            if box_gap(region, nearest) <= limit:
                pair = tuple(sorted((region, nearest), key=lambda r: r.index))
                pairs.setdefault((pair[0].region_id, pair[1].region_id), merge_regions(pair))
    return [pairs[key] for key in sorted(pairs)]
