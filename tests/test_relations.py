"""Tests of cora.data.relations on synthetic label maps (supplement Sec. 8; Fig. 2)."""
import numpy as np
import pytest

from cora.data.objects import extract_objects
from cora.data.relations import (
    ALL_RELATIONS, DEFAULT_THRESHOLDS, DEPTH_RELATIONS, RELATIONS, RegionGroup, anchor_groups, box_gap, merge_regions,
    region_members, relations_for, mask_gap, related_by_class, related_regions, relation_holds,
    resolve_thresholds,
)

NAMES = ["zero", "anchor", "thing", "ring"]
H, W = 100, 200


def blank():
    return np.full((H, W), 255, dtype=np.uint8)


def box(label, rows, cols, value):
    label[rows[0]:rows[1], cols[0]:cols[1]] = value


def objects_of(label):
    return extract_objects("synthetic", label, NAMES)


def cross_scene():
    """Anchor (class 1) in the middle with one 'thing' (class 2) far to each side."""
    label = blank()
    box(label, (40, 60), (90, 110), 1)  # anchor 1_1, centroid (99.5, 49.5)
    box(label, (45, 55), (10, 30), 2)  # 2_2 left   (ids follow the raster order of the first pixel)
    box(label, (45, 55), (150, 170), 2)  # 2_3 right
    box(label, (5, 15), (95, 105), 2)  # 2_1 above
    box(label, (85, 95), (95, 105), 2)  # 2_4 below
    return objects_of(label)


def test_relations_constant_lists_the_contract_vocabulary():
    assert RELATIONS == ("above", "below", "to the left of", "to the right of", "near", "around", "surrounding")


@pytest.mark.parametrize(
    "relation, expected",
    [("to the left of", "2_2"), ("to the right of", "2_3"), ("above", "2_1"), ("below", "2_4")],
)
def test_each_direction_selects_exactly_its_target(relation, expected):
    objs = cross_scene()
    anchor = objs.get("1_1")
    found = related_regions(objs, anchor, 2, relation)
    assert [r.region_id for r in found] == [expected]
    for region in objs.of_class(2):
        assert relation_holds(region, anchor, relation, (H, W)) == (region.region_id == expected)


def test_a_region_satisfies_at_most_one_direction():
    objs = cross_scene()
    anchor = objs.get("1_1")
    for region in objs.of_class(2):
        holding = [rel for rel in RELATIONS[:4] if relation_holds(region, anchor, rel, (H, W))]
        assert len(holding) == 1


def test_image_frame_left_means_smaller_x():
    label = blank()
    box(label, (40, 50), (150, 160), 1)
    box(label, (40, 50), (20, 30), 2)
    objs = objects_of(label)
    assert [r.region_id for r in related_regions(objs, objs.get("1_1"), 2, "to the left of")] == ["2_1"]
    assert related_regions(objs, objs.get("1_1"), 2, "to the right of") == []


def test_anchor_is_never_a_target_even_for_its_own_class():
    label = blank()
    box(label, (40, 50), (150, 160), 1)
    box(label, (40, 50), (20, 30), 1)
    objs = objects_of(label)
    anchor = objs.get("1_2")  # raster order: the left one is 1_1
    assert anchor.bbox[0] == 150
    found = related_regions(objs, anchor, 1, "to the left of")
    assert [r.region_id for r in found] == ["1_1"]
    assert not relation_holds(anchor, anchor, "to the left of", (H, W))
    assert related_regions(objs, anchor, 1, "near") == []


def test_direction_needs_the_dominant_axis_and_a_clear_margin():
    label = blank()
    box(label, (40, 50), (100, 110), 1)  # anchor centroid (104.5, 44.5)
    box(label, (10, 20), (140, 150), 2)  # dx = 40, dy = -30 -> horizontal dominates: right, not above
    box(label, (48, 58), (108, 114), 3)  # dx = 6 > margin 0.02*200 = 4, dy = 8: vertical dominates -> below
    objs = objects_of(label)
    anchor = objs.get("1_1")
    assert [r.region_id for r in related_regions(objs, anchor, 2, "to the right of")] == ["2_1"]
    assert related_regions(objs, anchor, 2, "above") == []
    assert [r.region_id for r in related_regions(objs, anchor, 3, "below")] == ["3_1"]
    # a 10 % margin (20 px horizontally) removes the target that sits only 6 px aside
    assert related_regions(objs, anchor, 3, "to the right of", direction_margin_frac=0.1) == []


def test_box_may_not_reach_past_the_anchor_centre():
    label = blank()
    box(label, (40, 50), (100, 110), 1)  # centroid x = 104.5
    box(label, (20, 30), (60, 106), 2)  # centroid x 82.5 is left of the anchor, but the box ends at x = 106 > 104.5
    objs = objects_of(label)
    assert related_regions(objs, objs.get("1_1"), 2, "to the left of") == []


def test_min_area_drops_small_targets_and_is_validated():
    label = blank()
    box(label, (40, 50), (150, 160), 1)
    box(label, (40, 50), (20, 30), 2)  # 100 px
    box(label, (60, 62), (20, 22), 2)  # 4 px
    objs = objects_of(label)
    anchor = objs.get("1_1")
    assert len(related_regions(objs, anchor, 2, "to the left of")) == 2
    assert [r.area for r in related_regions(objs, anchor, 2, "to the left of", min_area=10)] == [100]
    assert not relation_holds(objs.get("2_2"), anchor, "to the left of", (H, W), min_area=10)
    with pytest.raises(ValueError, match="min_area"):
        resolve_thresholds({"min_area": 0})


def distance_scene(third=True):
    """Anchor 20x20 with three 4x4 'things' 9 px away on the left, right and top."""
    label = blank()
    box(label, (40, 60), (90, 110), 1)
    box(label, (48, 52), (78, 82), 2)
    box(label, (48, 52), (118, 122), 2)
    if third:
        box(label, (28, 32), (98, 102), 2)
    return objects_of(label)


def test_near_uses_the_distance_to_the_anchor_boundary():
    objs = distance_scene()
    anchor = objs.get("1_1")
    diag = float(np.hypot(H, W))
    assert 9 < 0.05 * diag  # the things are within the default near radius (11.2 px)
    assert len(related_regions(objs, anchor, 2, "near")) == 3
    assert related_regions(objs, anchor, 2, "near", near_frac=0.03) == []  # 6.7 px radius
    far = blank()
    box(far, (40, 60), (90, 110), 1)
    box(far, (48, 52), (140, 144), 2)
    far_objs = objects_of(far)
    assert related_regions(far_objs, far_objs.get("1_1"), 2, "near") == []
    assert len(related_regions(far_objs, far_objs.get("1_1"), 2, "near", near_frac=0.2)) == 1


def test_mask_gap_and_box_gap_agree_on_rectangles_and_box_is_a_lower_bound():
    objs = distance_scene()
    anchor, left = objs.get("1_1"), objs.get("2_2")
    assert box_gap(left, anchor) == pytest.approx(9.0)
    assert mask_gap(objs.mask(["2_2"]), objs.mask(["1_1"])) == pytest.approx(9.0)
    # an L-shaped region: the box gap is smaller than the real gap
    label = blank()
    box(label, (40, 50), (90, 100), 1)
    box(label, (60, 80), (60, 62), 2)
    box(label, (78, 80), (60, 100), 2)
    lshape = objects_of(label)
    assert box_gap(lshape.get("2_1"), lshape.get("1_1")) < mask_gap(lshape.mask(["2_1"]), lshape.mask(["1_1"]))
    with pytest.raises(ValueError, match="non-empty"):
        mask_gap(np.zeros((3, 3), bool), np.ones((3, 3), bool))


def test_relation_holds_with_masks_matches_related_regions():
    objs = distance_scene()
    anchor = objs.get("1_1")
    for region in objs.of_class(2):
        assert relation_holds(region, anchor, "near", (H, W), target_mask=objs.mask([region.region_id]),
                              anchor_mask=objs.mask(["1_1"]))
    with pytest.raises(ValueError, match="both target_mask and anchor_mask"):
        relation_holds(objs.get("2_1"), anchor, "near", (H, W), target_mask=objs.mask(["2_1"]))


def test_around_needs_members_in_enough_compass_sectors():
    objs = distance_scene(third=True)
    anchor = objs.get("1_1")
    assert len(related_regions(objs, anchor, 2, "around")) == 3  # left, right, top = three sectors
    assert related_regions(objs, anchor, 2, "around", around_min_octants=4) == []
    two = distance_scene(third=False)  # left and right only
    assert related_regions(two, two.get("1_1"), 2, "around") == []
    assert len(related_regions(two, two.get("1_1"), 2, "near")) == 2
    assert len(related_regions(two, two.get("1_1"), 2, "around", around_min_octants=2)) == 2


def test_around_radius_is_larger_than_the_near_radius():
    label = blank()
    box(label, (40, 60), (90, 110), 1)
    for rows, cols in (((48, 52), (126, 130)), ((48, 52), (70, 74)), ((20, 24), (98, 102))):  # 17 px away on three sides
        box(label, rows, cols, 2)
    objs = objects_of(label)
    anchor = objs.get("1_1")
    assert related_regions(objs, anchor, 2, "near") == []  # beyond 11.2 px
    assert len(related_regions(objs, anchor, 2, "around")) == 3  # within 22.4 px


def ring_scene(full=True):
    label = blank()
    box(label, (25, 75), (75, 125), 3)  # annulus class, 15 px thick around the anchor
    box(label, (40, 60), (90, 110), 1)
    if not full:
        label[25:75, 75:110] = 255  # keep only the strip to the right of the anchor
        box(label, (40, 60), (90, 110), 1)
    return objects_of(label)


def test_surrounding_needs_the_members_to_cover_the_anchor_ring():
    objs = ring_scene(full=True)
    anchor = objs.get("1_1")
    assert [r.region_id for r in related_regions(objs, anchor, 3, "surrounding")] == ["3_1"]
    half = ring_scene(full=False)
    anchor = half.get("1_1")
    assert [r.region_id for r in related_regions(half, anchor, 3, "near")] == ["3_1"]
    assert related_regions(half, anchor, 3, "surrounding") == []
    assert [r.region_id for r in related_regions(half, anchor, 3, "surrounding", surround_cover=0.2)] == ["3_1"]


def test_group_conditions_are_checked_per_class_not_across_classes():
    label = blank()
    box(label, (40, 60), (90, 110), 1)
    box(label, (48, 52), (78, 82), 2)  # class 2 only on the left ...
    box(label, (48, 52), (118, 122), 3)  # ... class 3 on the right and at the top: no class has three sectors alone
    box(label, (28, 32), (98, 102), 3)
    objs = objects_of(label)
    anchor = objs.get("1_1")
    assert len(related_by_class(objs, anchor, "near")) == 2  # both classes are near
    assert related_by_class(objs, anchor, "around") == {}
    # one half of the ring from class 2 and the other half from class 3: each covers 50 %, not the 60 % asked for
    ring_left = blank()
    box(ring_left, (25, 75), (75, 125), 3)
    ring_left[25:75, 100:125] = 2
    box(ring_left, (40, 60), (90, 110), 1)
    both = objects_of(ring_left)
    assert related_by_class(both, both.get("1_1"), "surrounding", surround_cover=0.6) == {}
    assert sorted(related_by_class(both, both.get("1_1"), "surrounding", surround_cover=0.5)) == [2, 3]
    assert sorted(related_by_class(both, both.get("1_1"), "near")) == [2, 3]


def test_ring_beyond_the_image_border_does_not_count_against_coverage():
    label = blank()
    box(label, (0, 40), (0, 40), 3)
    box(label, (0, 20), (0, 20), 1)  # the anchor sits in the corner; the ring outside the image is not counted
    objs = objects_of(label)
    assert [r.region_id for r in related_regions(objs, objs.get("1_1"), 3, "surrounding")] == ["3_1"]


def test_related_by_class_groups_every_class():
    objs = cross_scene()
    grouped = related_by_class(objs, objs.get("1_1"), "to the left of")
    assert list(grouped) == [2] and [r.region_id for r in grouped[2]] == ["2_2"]


def test_unknown_relation_and_threshold_raise():
    objs = cross_scene()
    anchor = objs.get("1_1")
    with pytest.raises(ValueError, match="unknown relation"):
        related_regions(objs, anchor, 2, "beside")
    with pytest.raises(ValueError, match="unknown relation"):
        relation_holds(objs.get("2_1"), anchor, "beside", (H, W))
    with pytest.raises(ValueError, match="unknown relation threshold"):
        related_regions(objs, anchor, 2, "near", near=3)
    with pytest.raises(ValueError, match="around_min_octants"):
        resolve_thresholds({"around_min_octants": 9})
    with pytest.raises(ValueError, match="surround_cover"):
        resolve_thresholds({"surround_cover": 0.0})
    with pytest.raises(ValueError, match="near_frac"):
        resolve_thresholds({"near_frac": 1.5})


def test_defaults_are_documented_values():
    assert DEFAULT_THRESHOLDS == {
        "direction_margin_frac": 0.02, "near_frac": 0.05, "around_frac": 0.10, "around_min_octants": 3,
        "surround_cover": 0.5, "min_area": 1, "depth_margin_frac": 0.03, "depth_overlap_frac": 0.25, "group_frac": 0.03,
    }


def test_results_are_deterministic_and_in_index_order():
    objs = cross_scene()
    first = related_by_class(objs, objs.get("1_1"), "near", near_frac=0.5)
    second = related_by_class(objs, objs.get("1_1"), "near", near_frac=0.5)
    assert [r.region_id for r in first[2]] == [r.region_id for r in second[2]]
    assert [r.index for r in first[2]] == sorted(r.index for r in first[2])


# ----------------------------------------------------------------------------------------------
# Depth relations ("in front of" / "behind") and group anchors
# ----------------------------------------------------------------------------------------------
CITY_NAMES = ["road", "car", "person", "sky"]


def city_objects(label):
    return extract_objects("synthetic", label, CITY_NAMES)


def depth_scene():
    """A far car (bottom edge y=40) and a near car (y=80) on one line of sight; a third car off to the side."""
    label = blank()
    box(label, (30, 40), (90, 120), 1)  # far car, 1_1 (first in raster order)
    box(label, (60, 80), (80, 130), 1)  # near car, 1_2
    box(label, (60, 80), (10, 40), 1)  # side car 1_3 ... raster order puts it before 1_2
    return city_objects(label)


def test_depth_relations_are_cityscapes_only_in_the_default_vocabulary():
    assert DEPTH_RELATIONS == ("in front of", "behind") and ALL_RELATIONS == RELATIONS + DEPTH_RELATIONS
    assert relations_for("cityscapes") == ALL_RELATIONS and relations_for("another dataset") == RELATIONS


def test_in_front_of_and_behind_follow_the_bottom_edge_on_one_line_of_sight():
    objs = depth_scene()
    far = next(r for r in objs.of_class(1) if r.bbox[3] == 40)
    near = next(r for r in objs.of_class(1) if r.bbox[3] == 80 and r.bbox[0] == 80)
    side = next(r for r in objs.of_class(1) if r.bbox[0] == 10)
    assert relation_holds(near, far, "in front of", (H, W)) and not relation_holds(near, far, "behind", (H, W))
    assert relation_holds(far, near, "behind", (H, W)) and not relation_holds(far, near, "in front of", (H, W))
    assert not relation_holds(side, far, "in front of", (H, W))  # no horizontal overlap: side by side, not in line
    assert [r.region_id for r in related_regions(objs, far, 1, "in front of")] == [near.region_id]
    assert [r.region_id for r in related_regions(objs, near, 1, "behind")] == [far.region_id]


def test_depth_needs_a_margin_overlap_and_instance_classes():
    label = blank()
    box(label, (30, 40), (90, 120), 1)
    box(label, (41, 42), (95, 125), 1)  # a separate region (row 40 is empty) whose bottom edge is only 2 px lower (< 3 % of 100)
    objs = city_objects(label)
    a, b = objs.of_class(1)
    assert not relation_holds(b, a, "in front of", (H, W)) and not relation_holds(b, a, "behind", (H, W))
    assert relation_holds(b, a, "in front of", (H, W), depth_margin_frac=0.01)
    # a stuff class (road) never takes part: its bottom edge says nothing about distance
    label = blank()
    box(label, (30, 40), (90, 120), 1)
    box(label, (60, 90), (60, 150), 0)
    stuff = city_objects(label)
    assert not relation_holds(stuff.of_class(0)[0], stuff.of_class(1)[0], "in front of", (H, W))
    assert related_regions(stuff, stuff.of_class(1)[0], 0, "in front of") == []
    with pytest.raises(ValueError, match="depth_overlap_frac"):
        resolve_thresholds({"depth_overlap_frac": 0})


def test_depth_against_a_group_anchor_must_hold_for_both_members():
    """A car between two persons in depth is neither in front of nor behind the pair."""
    from cora.data.relations import merge_regions

    label = blank()
    box(label, (30, 50), (40, 50), 2)   # far person, bottom edge 50
    box(label, (40, 70), (52, 62), 2)   # near person, bottom edge 70
    box(label, (55, 60), (40, 50), 1)   # a car between them in depth (bottom edge 60), below the far person
    box(label, (80, 90), (44, 60), 1)   # a second car in front of both (bottom edge 90)
    objs = city_objects(label)
    far, near = sorted(objs.of_class(2), key=lambda r: r.bbox[3])
    middle, front = sorted(objs.of_class(1), key=lambda r: r.bbox[3])
    assert (far.bbox[3], near.bbox[3], middle.bbox[3], front.bbox[3]) == (50, 70, 60, 90)
    pair = merge_regions([far, near])
    assert relation_holds(middle, far, "in front of", (H, W))
    assert not relation_holds(middle, pair, "behind", (H, W)) and not relation_holds(middle, pair, "in front of", (H, W))
    assert relation_holds(front, pair, "in front of", (H, W)) and not relation_holds(front, pair, "behind", (H, W))


def test_depth_never_fires_for_riders():
    """A rider sits on a vehicle: its bottom edge is not on the ground, so the ground-plane cue does not apply."""
    label = blank()
    box(label, (30, 40), (90, 120), 1)  # rider
    box(label, (60, 80), (80, 130), 2)  # the vehicle below it, on the same line of sight
    objs = extract_objects("synthetic", label, ["road", "rider", "bicycle", "sky"])
    rider, bike = objs.of_class(1)[0], objs.of_class(2)[0]
    for relation in ("in front of", "behind"):
        assert not relation_holds(rider, bike, relation, (H, W)) and not relation_holds(bike, rider, relation, (H, W))
    assert relation_holds(rider, bike, "above", (H, W))  # the 2D relations are unaffected


def test_depth_never_fires_on_classes_without_an_instance_name():
    objs = cross_scene()  # classes named "anchor", "thing", "ring"
    for region in objs.of_class(2):
        assert not relation_holds(region, objs.get("1_1"), "in front of", (H, W))
        assert not relation_holds(region, objs.get("1_1"), "behind", (H, W))


def group_scene():
    label = blank()
    box(label, (40, 60), (20, 30), 2)  # person 2_1
    box(label, (40, 60), (34, 44), 2)  # person 2_2, 4 px from 2_1
    box(label, (40, 60), (150, 160), 2)  # person 2_3, alone
    box(label, (10, 20), (100, 110), 1)  # car 1_1
    return city_objects(label)


def test_anchor_groups_pair_nearest_same_class_neighbours_within_group_frac():
    objs = group_scene()
    groups = anchor_groups(objs)
    assert [g.region_id for g in groups] == ["2_1+2_2"]
    group = groups[0]
    assert isinstance(group, RegionGroup) and group.label_name == "person"
    assert group.area == sum(m.area for m in group.members) and group.bbox == (20, 40, 44, 60)
    assert group.centroid == pytest.approx((31.5, 49.5)) and region_members(group) == group.members
    assert anchor_groups(objs, group_frac=0.5)  # a looser limit pairs the loner too
    assert anchor_groups(objs, min_member_area=10_000) == []
    assert anchor_groups(objs) == anchor_groups(objs)  # deterministic


def test_merge_regions_rejects_bad_input():
    objs = group_scene()
    with pytest.raises(ValueError, match="at least two"):
        merge_regions([objs.get("2_1")])
    with pytest.raises(ValueError, match="share one class"):
        merge_regions([objs.get("2_1"), objs.get("1_1")])


def test_group_anchor_excludes_every_member_and_measures_from_the_union():
    objs = group_scene()
    group = anchor_groups(objs)[0]
    found = related_by_class(objs, group, "to the right of")
    assert [r.region_id for r in found[2]] == ["2_3"]  # neither member is a target of its own group
    assert 2 in related_by_class(objs, group, "near", near_frac=0.9)
    assert all(r.region_id not in ("2_1", "2_2") for rs in related_by_class(objs, group, "near", near_frac=0.9).values() for r in rs)
    assert not relation_holds(objs.get("2_2"), group, "near", (H, W))
    assert box_gap(objs.get("2_3"), group) == box_gap(objs.get("2_3"), objs.get("2_2"))  # member boxes, not the union box
