"""Tests of cora.data.objects (supplement Sec. 4: connected-component objects) and tools/build_objects.py."""
import importlib.util
import json
import multiprocessing
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy import ndimage

from cora import constants as C
from cora.config import Config
from cora.data import objects as O
from cora.data.objects import (
    ImageObjects, Region, ensure_objects, extract_objects, load_objects, mask_polygons, polygons_to_mask, save_objects,
)

NAMES = ["zero", "one", "two", "three", "four"]
REPO = Path(__file__).resolve().parents[1]


def label_map(height=12, width=16, fill=255):
    return np.full((height, width), fill, dtype=np.uint8)


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # worker processes pickle functions of the tool by module name
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------------------------
def test_class_split_into_components_and_region_ids():
    label = label_map()
    label[0:2, 10:13] = 3  # first in raster order although it is to the right
    label[5:8, 0:3] = 3
    label[9:11, 5:8] = 1
    objs = extract_objects("img", label, NAMES)
    assert [r.region_id for r in objs.regions] == ["1_1", "3_1", "3_2"]
    assert [r.index for r in objs.regions] == [1, 2, 3]
    first, second = objs.get("3_1"), objs.get("3_2")
    assert first.bbox == (10, 0, 13, 2) and first.area == 6 and first.label_name == "three"
    assert second.bbox == (0, 5, 3, 8) and second.area == 9
    assert first.centroid == pytest.approx((11.0, 0.5))
    assert objs.region_map.dtype == np.uint16 and objs.region_map.shape == label.shape
    assert objs.region_map[0, 10] == first.index and objs.region_map[6, 1] == second.index
    assert objs.region_map[4, 4] == 0
    assert [r.region_id for r in objs.of_class(3)] == ["3_1", "3_2"] and objs.of_class(2) == []


def test_ignore_label_gets_no_region():
    label = label_map()
    label[2:4, 2:4] = 2
    objs = extract_objects("img", label, NAMES)
    assert len(objs.regions) == 1 and int((objs.region_map > 0).sum()) == 4
    custom = np.full((6, 6), 9, dtype=np.int32)
    custom[1:3, 1:3] = 0
    assert [r.region_id for r in extract_objects("img", custom, NAMES, ignore_label=9).regions] == ["0_1"]
    assert extract_objects("img", label_map(), NAMES).regions == []


def test_diagonal_touching_depends_on_connectivity():
    label = label_map(6, 6)
    label[1, 1] = label[2, 2] = 4
    assert [r.region_id for r in extract_objects("i", label, NAMES, connectivity=8).regions] == ["4_1"]
    four = extract_objects("i", label, NAMES, connectivity=4)
    assert [r.region_id for r in four.regions] == ["4_1", "4_2"]
    assert four.get("4_1").bbox == (1, 1, 2, 2)  # raster order: (1, 1) comes before (2, 2)


def test_min_area_filters_and_renumbers_contiguously():
    label = label_map(10, 20)
    label[0, 0] = 2  # area 1
    label[0, 5:8] = 2  # area 3
    label[3:5, 0:3] = 2  # area 6
    label[8, 18] = 2  # area 1
    kept = extract_objects("i", label, NAMES, min_area=3)
    assert [(r.region_id, r.area) for r in kept.regions] == [("2_1", 3), ("2_2", 6)]
    assert kept.region_map[0, 0] == 0 and kept.region_map[8, 18] == 0
    assert kept.min_area == 3 and kept.connectivity == 8
    assert len(extract_objects("i", label, NAMES, min_area=1).regions) == 4


def test_invalid_inputs():
    label = label_map()
    with pytest.raises(ValueError, match="connectivity"):
        extract_objects("i", label, NAMES, connectivity=6)
    with pytest.raises(ValueError, match="min_area"):
        extract_objects("i", label, NAMES, min_area=0)
    with pytest.raises(ValueError, match="2-D integer"):
        extract_objects("i", label.astype(np.float32), NAMES)
    with pytest.raises(ValueError, match="outside"):
        extract_objects("i", np.full((3, 3), 7, dtype=np.uint8), NAMES)


def test_too_many_regions_for_uint16():
    checker = np.indices((400, 400)).sum(0) % 2 == 0
    label = np.where(checker, 0, 255).astype(np.uint8)
    with pytest.raises(ValueError, match="uint16"):
        extract_objects("i", label, NAMES, connectivity=4)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_random_masks_agree_with_scipy_and_raster_order(connectivity):
    """The labelling really yields raster order of the first pixel; areas, boxes and centroids are exact."""
    rng = np.random.default_rng(0)
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    for _ in range(150):
        h, w = (int(v) for v in rng.integers(4, 40, 2))
        label = np.where(rng.random((h, w)) < rng.uniform(0.15, 0.7), 1, 255).astype(np.uint8)
        label[rng.random((h, w)) < 0.1] = 3  # a second class interleaved
        objs = extract_objects("r", label, NAMES, connectivity=connectivity)
        for class_id in (1, 3):
            expected = ndimage.label(label == class_id, structure)[1]
            regions = objs.of_class(class_id)
            assert len(regions) == expected
            assert [r.region_id for r in regions] == [C.region_id(class_id, k) for k in range(1, expected + 1)]
            firsts = [int(np.flatnonzero(objs.region_map.ravel() == r.index)[0]) for r in regions]
            assert firsts == sorted(firsts)
            union = np.zeros((h, w), bool)
            for r in regions:
                comp = objs.region_map == r.index
                ys, xs = np.nonzero(comp)
                assert r.area == comp.sum() == len(ys)
                assert r.bbox == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                assert r.centroid == pytest.approx((xs.mean(), ys.mean()))
                union |= comp
            assert np.array_equal(union, label == class_id)


def test_raster_order_helper_does_not_rely_on_opencv_order():
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, 7:9] = True
    mask[2:4, 0:2] = True
    mask[4, 4] = True
    labels, order, _, _ = O._components_in_raster_order(mask, 8)
    flat = labels.ravel()
    firsts = [int(np.flatnonzero(flat == lab)[0]) for lab in order]
    assert len(order) == 3 and firsts == sorted(firsts) == [7, 18, 40]


def test_mask_get_and_errors():
    label = label_map()
    label[0:2, 0:2] = 1
    label[5:7, 5:7] = 1
    label[9:11, 9:11] = 2
    objs = extract_objects("some/image", label, NAMES)
    assert objs.mask(["1_1"]).sum() == 4
    assert np.array_equal(objs.mask(["1_1", "1_2"]), label == 1)
    assert not objs.mask([]).any() and objs.mask([]).dtype == bool
    with pytest.raises(KeyError, match="some/image.*9_9"):
        objs.get("9_9")
    with pytest.raises(KeyError):
        objs.mask(["1_1", "1_7"])


# ----------------------------------------------------------------------------------------------
# Polygons
# ----------------------------------------------------------------------------------------------
def test_polygons_reproduce_random_components():
    rng = np.random.default_rng(1)
    for _ in range(80):
        h, w = (int(v) for v in rng.integers(6, 40, 2))
        blob = ndimage.binary_dilation(rng.random((h, w)) < 0.05, iterations=int(rng.integers(1, 3)))
        for comp_label in range(1, ndimage.label(blob, np.ones((3, 3)))[1] + 1):
            comp = ndimage.label(blob, np.ones((3, 3)))[0] == comp_label
            polys = mask_polygons(comp)
            assert len(polys) == 1
            assert np.array_equal(polygons_to_mask(polys, comp.shape), ndimage.binary_fill_holes(comp))


def test_polygons_several_components_and_format():
    mask = np.zeros((10, 12), bool)
    mask[5:8, 1:4] = True
    mask[1:3, 8:11] = True
    polys = mask_polygons(mask)
    assert len(polys) == 2
    assert min(polys[0], key=lambda p: (p[1], p[0]))[1] == 1  # ordered by topmost point
    assert all(isinstance(v, int) for poly in polys for pt in poly for v in pt)
    assert np.array_equal(polygons_to_mask(polys, mask.shape), mask)
    assert mask_polygons(np.zeros((4, 4), bool)) == []
    single = np.zeros((4, 4), bool)
    single[2, 1] = True
    assert polygons_to_mask(mask_polygons(single), single.shape).tolist() == single.tolist()


def test_polygon_holes_option():
    ring = np.zeros((20, 20), bool)
    ring[2:18, 2:18] = True
    ring[6:14, 6:14] = False
    ring[9:11, 9:11] = True  # island inside the hole
    assert not np.array_equal(polygons_to_mask(mask_polygons(ring), ring.shape), ring)
    with_holes = mask_polygons(ring, include_holes=True)
    assert len(with_holes) == 3
    assert np.array_equal(polygons_to_mask(with_holes, ring.shape), ring)


def test_default_polygons_omit_components_nested_in_a_hole():
    ring = np.zeros((9, 9), bool)
    ring[1:8, 1:8] = True
    ring[3:6, 3:6] = False
    ring[4, 4] = True
    assert len(mask_polygons(ring)) == 1  # documented: the island inside the hole is not an outermost contour
    assert "omitted" in mask_polygons.__doc__
    assert len(mask_polygons(ring, include_holes=True)) == 3


def test_large_tolerance_never_yields_degenerate_polygons():
    square = np.zeros((30, 30), bool)
    square[5:25, 5:25] = True
    exact = mask_polygons(square)
    for polygon in mask_polygons(square, tolerance=50):
        assert len(polygon) >= 3
    assert mask_polygons(square, tolerance=50) == exact  # collapse -> exact contour kept
    assert np.array_equal(polygons_to_mask(mask_polygons(square, tolerance=50), square.shape), square)
    single = np.zeros((5, 5), bool)
    single[2, 2] = True
    assert mask_polygons(single, tolerance=3) == [[[2, 2]]]  # a one-pixel contour has no more vertices to give


def test_polygon_simplification():
    yy, xx = np.mgrid[:80, :80]
    disc = (yy - 40) ** 2 + (xx - 40) ** 2 <= 30**2
    exact, simple = mask_polygons(disc)[0], mask_polygons(disc, tolerance=1.5)[0]
    assert len(simple) < len(exact) / 3
    approx = polygons_to_mask([simple], disc.shape)
    assert (approx & disc).sum() / (approx | disc).sum() > 0.95
    with pytest.raises(ValueError, match="tolerance"):
        mask_polygons(disc, tolerance=-1)
    with pytest.raises(ValueError, match="2-D"):
        mask_polygons(disc[None])


def test_region_polygons_of_extracted_objects():
    label = label_map(30, 30)
    label[3:12, 4:20] = 2
    label[20:27, 3:9] = 2
    objs = extract_objects("p", label, NAMES)
    for region in objs.regions:
        mask = objs.mask([region.region_id])
        assert np.array_equal(polygons_to_mask(mask_polygons(mask), mask.shape), mask)


# ----------------------------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------------------------
def test_save_load_round_trip_with_nested_ids(tmp_path):
    label = label_map(20, 30)
    label[1:5, 1:5] = 0
    label[10:15, 20:28] = 2
    label[16:19, 2:6] = 2
    objs = extract_objects("train/aachen/aachen_000000_000019", label, NAMES, connectivity=4, min_area=2)
    save_objects(objs, tmp_path)
    assert (tmp_path / "train/aachen/aachen_000000_000019.png").is_file()
    assert (tmp_path / "train/aachen/aachen_000000_000019.json").is_file()
    assert not [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]
    back = load_objects(tmp_path, objs.image_id)
    assert (back.image_id, back.height, back.width) == (objs.image_id, 20, 30)
    assert (back.connectivity, back.min_area) == (4, 2)
    assert np.array_equal(back.region_map, objs.region_map) and back.region_map.dtype == np.uint16
    assert back.regions == objs.regions
    assert isinstance(back.regions[0].bbox, tuple) and isinstance(back.regions[0].centroid, tuple)
    save_objects(objs, tmp_path)  # overwriting is allowed
    assert O.objects_exist(tmp_path, objs.image_id) and not O.objects_exist(tmp_path, "other")


def test_mismatched_png_and_json_are_detected(tmp_path):
    first = extract_objects("a", label_map(10, 10), NAMES)
    other_label = label_map(10, 10)
    other_label[0:3, 0:3] = 2
    second = extract_objects("a", other_label, NAMES)
    assert not np.array_equal(first.region_map, second.region_map)
    save_objects(first, tmp_path)
    json_old = (tmp_path / "a.json").read_bytes()
    save_objects(second, tmp_path)
    (tmp_path / "a.json").write_bytes(json_old)  # new PNG, old JSON: what a reader sees during --force
    with pytest.raises(ValueError, match="different writes"):
        load_objects(tmp_path, "a")


def test_region_ids_above_255_survive_png(tmp_path):
    region_map = np.zeros((4, 4), np.uint16)
    region_map[0, 0], region_map[3, 3] = 300, 40000
    regions = [Region(i, C.region_id(0, i), 0, "zero", 0, (0, 0, 1, 1), (0.0, 0.0)) for i in range(1, 40001)]
    objs = ImageObjects("big", 4, 4, region_map, regions)
    save_objects(objs, tmp_path)
    assert np.array_equal(load_objects(tmp_path, "big").region_map, region_map)


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_objects(tmp_path, "missing")
    objs = extract_objects("a/b", label_map(), NAMES)
    save_objects(objs, tmp_path)
    (tmp_path / "a/b.png").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="uint16"):
        load_objects(tmp_path, "a/b")
    for bad in ("", "/abs", "a/../b", "a//b", "a\\b"):
        with pytest.raises(ValueError, match="invalid image id"):
            save_objects(ImageObjects(bad, 2, 2, np.zeros((2, 2), np.uint16), []), tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        ImageObjects("x", 3, 3, np.zeros((2, 2), np.uint16), [])


class FakeAdapter:
    name = "fake"
    class_names = tuple(NAMES)

    def __init__(self):
        self.calls = 0
        self.label = label_map(16, 16)
        self.label[2:5, 2:5] = 1
        self.label[8:12, 8:12] = 1
        self.label[0:2, 12:16] = 4

    def load_label(self, image_id):
        self.calls += 1
        return self.label


def make_cfg(tmp_path, **data):
    cfg = Config()
    cfg.paths.data_dir = str(tmp_path / "data")
    for key, value in data.items():
        setattr(cfg.data, key, value)
    return cfg


def test_ensure_objects_extracts_once_and_checks_parameters(tmp_path):
    cfg, adapter = make_cfg(tmp_path), FakeAdapter()
    first = ensure_objects(cfg, adapter, "train", "train/c/x")
    assert adapter.calls == 1 and [r.region_id for r in first.regions] == ["1_1", "1_2", "4_1"]
    assert (cfg.objects_dir("train") / "train/c/x.png").is_file()
    again = ensure_objects(cfg, adapter, "train", "train/c/x")
    assert adapter.calls == 1 and again.regions == first.regions
    assert not (cfg.objects_dir("val") / "train/c/x.png").exists()
    with pytest.raises(ValueError, match="connectivity=8"):
        ensure_objects(make_cfg(tmp_path, connectivity=4), adapter, "train", "train/c/x")
    with pytest.raises(ValueError, match="min_area=1"):
        ensure_objects(make_cfg(tmp_path, min_region_area=5), adapter, "train", "train/c/x")


# ----------------------------------------------------------------------------------------------
# tools/build_objects.py on a synthetic Cityscapes folder
# ----------------------------------------------------------------------------------------------
def write_synthetic_cityscapes(root: Path, n_per_split=3):
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for k in range(n_per_split):
            stem = f"town_{k:06d}_000019"
            (root / "leftImg8bit" / split / "town").mkdir(parents=True, exist_ok=True)
            (root / "gtFine" / split / "town").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / "leftImg8bit" / split / "town" / f"{stem}_leftImg8bit.png"),
                        rng.integers(0, 255, (16, 32, 3), dtype=np.uint8))
            label_ids = np.zeros((16, 32), np.uint8)
            label_ids[0:8, :] = 23  # sky
            label_ids[10:14, 2:6] = 24  # person
            label_ids[10:14, 10:14 + k] = 24
            label_ids[12:15, 20:30] = 26  # car
            cv2.imwrite(str(root / "gtFine" / split / "town" / f"{stem}_gtFine_labelIds.png"), label_ids)


@pytest.mark.parametrize("workers", [1, 2])
def test_build_objects_tool(tmp_path, workers, capsys):
    if workers > 1 and multiprocessing.get_start_method() != "fork":
        pytest.skip("the tool is loaded from its file here, which worker processes that are not forked cannot import")
    root = tmp_path / "cs"
    write_synthetic_cityscapes(root)
    tool = load_tool("build_objects")
    report = tmp_path / "report.json"
    args = ["--split", "train", "--workers", str(workers), "--report", str(report),
            "--set", f"paths.cityscapes_root={root}", "--set", f"paths.data_dir={tmp_path / 'out'}"]
    assert tool.main(args) == 0
    out = capsys.readouterr().out
    assert "person" in out and "car" in out
    summary = json.loads(report.read_text())
    assert summary["images"] == 3
    assert summary["classes"]["person"]["regions"] == 6 and summary["classes"]["person"]["images"] == 3
    assert summary["classes"]["sky"]["regions"] == 3 and summary["classes"]["bicycle"]["regions"] == 0
    stored = sorted((tmp_path / "out/cityscapes/objects/train/train/town").glob("*.png"))
    assert len(stored) == 3
    objs = load_objects(tmp_path / "out/cityscapes/objects/train", "train/town/town_000000_000019")
    assert sorted(r.region_id for r in objs.regions) == ["10_1", "11_1", "11_2", "13_1"]
    # a second run reuses the files; --limit and --force work
    assert tool.main(args + ["--limit", "1"]) == 0
    assert tool.main(args + ["--force"]) == 0
    with pytest.raises(SystemExit):
        tool.main(["--set", f"paths.cityscapes_root={root}"])


def test_build_objects_rejects_contradictory_selectors(tmp_path):
    tool = load_tool("build_objects")
    base = ["--set", f"paths.cityscapes_root={tmp_path}"]
    with pytest.raises(SystemExit, match="cannot be combined"):
        tool.main(["--split", "val", "--partition", "1_30"] + base)
    with pytest.raises(SystemExit, match="only one"):
        tool.main(["--partition", "1_30", "--image-list", str(tmp_path / "ids.txt")] + base)


# ----------------------------------------------------------------------------------------------
# tools/mask2instruction.py: the four-step mask-to-instruction recipe (paper Sec. 1 contribution)
# ----------------------------------------------------------------------------------------------
def m2i_args(tmp_path, *extra):
    root = tmp_path / "cs"
    if not root.exists():
        write_synthetic_cityscapes(root)
    return ["--set", f"paths.cityscapes_root={root}", "--set", f"paths.data_dir={tmp_path / 'out'}", *extra]


def test_mask2instruction_plan_order_and_arguments(tmp_path):
    tool = load_tool("mask2instruction")
    plan = tool.build_plan(tool.parse_args(m2i_args(tmp_path, "--partition", "1_16", "--splits", "train", "val", "--limit", "4")))
    assert [(c.step, c.tool) for c in plan] == [
        ("objects", "build_objects"), ("objects", "build_objects"),
        ("rule-sets", "generate_rule_sets"), ("rule-sets", "generate_rule_sets"),
        ("rule-sets", "generate_rule_sets"), ("rule-sets", "generate_rule_sets"),
        ("query-db", "generate_rule_sets"),
    ]
    assert "--partition" in plan[0].argv and "1_16" in plan[0].argv and "--split" not in plan[0].argv
    assert plan[1].argv[-6:-4] == ("--split", "val") and plan[1].argv[-2:] == ("--limit", "4")
    assert "semantic" in plan[2].argv and "templated-conditional" in plan[3].argv
    assert plan[-1].argv[-1] == "query-db"
    assert plan[0].shell().startswith("python tools/build_objects.py ")


def test_mask2instruction_teacher_step(tmp_path):
    tool = load_tool("mask2instruction")
    args = tool.parse_args(m2i_args(tmp_path, "--partition", "1_16",
                                     "--teacher", "unimatch", "--checkpoint", "ck.pth", "--date", "2026-11-19", "--gpus", "7"))
    plan = tool.build_plan(args)
    pseudo = plan[-1]
    assert pseudo.step == "pseudo" and pseudo.tool == "make_pseudo_labels"
    assert {"--method", "unimatch", "--checkpoint", "ck.pth", "--gpus", "7", "--dataset", "cityscapes"} <= set(pseudo.argv)
    assert [c.step for c in plan].index("pseudo") > [c.step for c in plan].index("query-db")


@pytest.mark.parametrize("argv, message", [
    (["--splits", "train"], "needs --partition"),
    (["--partition", "1_16", "--teacher", "unimatch", "--date", "2026-11-19"], "needs --checkpoint"),
    (["--partition", "1_16", "--teacher", "unimatch", "--checkpoint", "c"], "--partition and --date"),
    (["--partition", "1_16", "--steps", "pseudo"], "needs --teacher"),
])
def test_mask2instruction_rejects_inconsistent_options(tmp_path, argv, message):
    tool = load_tool("mask2instruction")
    with pytest.raises(SystemExit, match=message):
        tool.build_plan(tool.parse_args(argv))


def test_mask2instruction_dry_run_runs_nothing(tmp_path, capsys):
    tool = load_tool("mask2instruction")
    calls = []
    assert tool.main(m2i_args(tmp_path, "--partition", "1_16", "--dry-run"), runner=calls.append) == 0
    assert not calls and "[objects]" in capsys.readouterr().out and not (tmp_path / "out").exists()


def test_mask2instruction_stops_at_first_failure(tmp_path):
    tool = load_tool("mask2instruction")
    seen = []
    def runner(command):
        seen.append(command.step)
        return 3 if command.step == "rule-sets" else 0
    with pytest.raises(SystemExit, match="step 'rule-sets'.*exit code 3"):
        tool.main(m2i_args(tmp_path, "--partition", "1_16"), runner=runner)
    assert seen == ["objects", "rule-sets"]


def test_mask2instruction_end_to_end_on_synthetic_cityscapes(tmp_path):
    tool = load_tool("mask2instruction")
    ids = tmp_path / "ids.txt"
    ids.write_text("train/town/town_000000_000019\ntrain/town/town_000001_000019\n")
    assert tool.main(m2i_args(tmp_path, "--image-list", str(ids), "--splits", "train", "val")) == 0
    out = tmp_path / "out" / "cityscapes"
    assert len(list((out / "objects" / "train" / "train" / "town").glob("*.png"))) == 2
    assert len(list((out / "objects" / "val" / "val" / "town").glob("*.png"))) == 3
    for split, n_images in (("train", 2), ("val", 3)):
        for name in ("semantic.jsonl", "conditional_templated.jsonl"):
            lines = (out / "instructions" / split / name).read_text().splitlines()
            assert lines and len({json.loads(line)["image_id"] for line in lines}) <= n_images
    assert (out / "query_db.json").is_file()
    with pytest.raises(FileNotFoundError, match="list file not found"):
        tool.main(m2i_args(tmp_path, "--image-list", str(tmp_path / "missing.txt"), "--steps", "objects"))


# ----------------------------------------------------------------------------------------------
# Real Cityscapes
# ----------------------------------------------------------------------------------------------
REAL_IDS = ["train/aachen/aachen_000000_000019", "train/bochum/bochum_000000_021070", "val/frankfurt/frankfurt_000001_062250"]


@pytest.mark.data
@pytest.mark.parametrize("image_id", REAL_IDS)
def test_real_cityscapes_objects(cityscapes_root, image_id, tmp_path):
    from cora.data.cityscapes import CityscapesAdapter

    adapter = CityscapesAdapter(cityscapes_root)
    label = adapter.load_label(image_id)
    for connectivity in (4, 8):
        objs = extract_objects(image_id, label, adapter.class_names, connectivity=connectivity, min_area=1)
        assert objs.region_map.shape == label.shape == (1024, 2048)
        for class_id in range(19):
            class_mask = label == class_id
            regions = objs.of_class(class_id)
            union = objs.mask([r.region_id for r in regions])
            assert np.array_equal(union, class_mask)
            assert sum(r.area for r in regions) == int(class_mask.sum())
            expected = ndimage.label(class_mask, ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2))[1]
            assert len(regions) == expected
            assert [r.region_id for r in regions] == [C.region_id(class_id, k) for k in range(1, expected + 1)]
        assert not (objs.region_map[label == C.IGNORE_LABEL] > 0).any()
    if (label == 11).any():
        assert objs.get("11_1").label_name == "person" and objs.get("11_1").label_id == 11
    # persistence through the config-driven path
    cfg = make_cfg(tmp_path)
    stored = ensure_objects(cfg, adapter, image_id.split("/")[0], image_id)
    assert np.array_equal(stored.region_map, objs.region_map)
    assert load_objects(cfg.objects_dir(image_id.split("/")[0]), image_id).regions == objs.regions


@pytest.mark.data
def test_real_images_have_person_regions(cityscapes_root):
    from cora.data.cityscapes import CityscapesAdapter

    adapter = CityscapesAdapter(cityscapes_root)
    found = set()
    for image_id in REAL_IDS:
        objs = extract_objects(image_id, adapter.load_label(image_id), adapter.class_names)
        found |= {r.region_id for r in objs.of_class(11)}
    assert "11_1" in found and all(rid.startswith("11_") for rid in found)


@pytest.mark.data
def test_real_polygons_reproduce_regions(cityscapes_root):
    from cora.data.cityscapes import CityscapesAdapter

    adapter = CityscapesAdapter(cityscapes_root)
    image_id = REAL_IDS[0]
    objs = extract_objects(image_id, adapter.load_label(image_id), adapter.class_names)
    checked = 0
    for region in objs.regions[:60]:
        x0, y0, x1, y1 = region.bbox
        mask = objs.region_map[y0:y1, x0:x1] == region.index
        polygons = mask_polygons(mask)
        assert len(polygons) == 1
        assert np.array_equal(polygons_to_mask(polygons, mask.shape), ndimage.binary_fill_holes(mask))
        with_holes = polygons_to_mask(mask_polygons(mask, include_holes=True), mask.shape)
        assert (with_holes ^ mask).sum() <= max(2, 0.001 * mask.sum())
        checked += 1
    assert checked > 10
