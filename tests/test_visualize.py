"""Tests of tools/visualize.py (qualitative panels as in main Fig. 6 and supplement Fig. 1) on the tiny model and synthetic data."""
import importlib.util
import json
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
import torch

from cora import constants as C
from cora.config import Config
from cora.data.batch import Conversation, Sample
from cora.data.collate import make_collate
from cora.data.datasets import EvalDataset
from cora.data.schema import InstructionRecord, ObjectRef, write_jsonl
from cora.data.transforms import anchor_box, anchor_crop, clip_preprocess, sam_preprocess
from tests.fixtures import tiny_config, tiny_model_and_tokenizer

REPO = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"tool_{name}", REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


viz = load_tool("visualize")


def count(panel: np.ndarray, predicate) -> int:
    r, g, b = (panel[..., i].astype(int) for i in range(3))
    return int(predicate(r, g, b).sum())


def is_red(r, g, b):
    return (r > 200) & (g < 60) & (b < 60)


def is_green(r, g, b):
    return (g > 150) & (r < 90) & (b < 90)


def gray_image(h=60, w=120) -> np.ndarray:
    return np.full((h, w, 3), 128, np.uint8)


def rect(h, w, y0, y1, x0, x1) -> np.ndarray:
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    return mask


# ----------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------
def test_overlay_tints_and_outlines_only_the_mask():
    image = gray_image(20, 20)
    mask = rect(20, 20, 5, 15, 5, 15)
    out = viz.overlay_mask(image, mask, viz.RED, alpha=0.5, thickness=1)
    assert out.dtype == np.uint8 and out.shape == image.shape
    assert tuple(out[10, 10]) == (192, 64, 64)  # the interior is a 50% blend of gray and red
    assert tuple(out[5, 5]) == viz.RED  # the outline has the pure colour
    assert np.array_equal(out[0:3], image[0:3]) and np.array_equal(out[:, 17:], image[:, 17:])
    assert np.array_equal(image, gray_image(20, 20))  # the input is not modified
    assert np.array_equal(viz.overlay_mask(image, np.zeros((20, 20), bool), viz.RED), image)
    with pytest.raises(ValueError, match="differ in size"):
        viz.overlay_mask(image, np.zeros((5, 5), bool), viz.RED)
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        viz.overlay_mask(image[..., 0], mask, viz.RED)


def test_mask_iou():
    target = rect(10, 10, 0, 5, 0, 10)
    assert viz.mask_iou(target, target) == 1.0
    assert viz.mask_iou(None, target) == 0.0
    assert viz.mask_iou(rect(10, 10, 0, 10, 0, 10), target) == pytest.approx(0.5)
    assert viz.mask_iou(None, np.zeros((10, 10), bool)) == 1.0


def test_wrap_words_respects_the_width():
    font = viz._font(16)
    words = ("Segment the cars which are parked on the left of the reference object <REF> please " * 3).split()
    lines = viz.wrap_words(words, font, 220)
    assert len(lines) > 1 and [w for line in lines for w in line] == words
    from PIL import Image, ImageDraw

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for line in lines:
        assert len(line) == 1 or probe.textlength(" ".join(line), font=font) <= 220
    assert viz.wrap_words([], font, 100) == []


def test_panel_colours_and_layout():
    image = gray_image()
    target, anchor, prediction = rect(60, 120, 10, 30, 10, 50), rect(60, 120, 35, 55, 70, 110), rect(60, 120, 10, 30, 20, 60)
    panel = viz.render_panel(image, "Segment the cars on the left of <REF>.", target, prediction, anchor, "title", panel_width=400)
    assert panel.dtype == np.uint8 and panel.ndim == 3 and panel.shape[2] == 3
    assert panel.shape[1] == 2 * 400 + 3 * viz.GAP
    assert panel.shape[0] > 200 + 2 * viz.GAP  # image of 200 px height plus the text block
    half = 400 + 2 * viz.GAP
    image_rows = slice(viz.GAP, viz.GAP + 200)
    left, right = panel[image_rows, :half], panel[image_rows, half:]
    assert count(left, is_red) > 0 and count(left, is_green) > 0  # target red, anchor green
    assert count(right, is_red) > 0 and count(right, is_green) == 0  # the prediction panel has no anchor
    # The prediction is shifted to the right of the target: compare where the red outline sits.
    red_cols_left = np.nonzero(is_red(*(left[..., i].astype(int) for i in range(3))).any(axis=0))[0]
    red_cols_right = np.nonzero(is_red(*(right[..., i].astype(int) for i in range(3))).any(axis=0))[0]
    assert red_cols_right.max() > red_cols_left.max()
    # Text below the images is drawn (dark pixels exist below the image area).
    below = panel[viz.GAP + 200 + 2 * viz.GAP:]
    assert (below.sum(axis=2) < 200).any()
    assert panel.shape[0] == viz.render_panel(image, "Segment the cars on the left of <REF>.", target, prediction, anchor, "title", 400).shape[0]


def test_ref_token_is_drawn_green_in_the_text_only():
    image = gray_image()
    target = rect(60, 120, 10, 30, 10, 50)

    def text_area(query: str) -> np.ndarray:
        panel = viz.render_panel(image, query, target, None, None, "", panel_width=400)
        return panel[viz.GAP + 200 + 2 * viz.GAP:]

    with_ref, without = text_area("Segment the cars on the left of <REF>."), text_area("Segment the cars on the left.")
    assert count(with_ref, lambda r, g, b: (g - r > 50) & (g - b > 50)) > 0
    assert count(without, lambda r, g, b: (g - r > 50) & (g - b > 50)) == 0


def test_missing_prediction_leaves_the_image_untouched_and_says_so():
    image, target = gray_image(), rect(60, 120, 10, 30, 10, 50)
    panel = viz.render_panel(image, "Segment the cars.", target, None, None, "", panel_width=400)
    half = 400 + 2 * viz.GAP
    right = panel[viz.GAP:viz.GAP + 200, half:half + 400]
    assert (right == 128).all()
    caption_zone = panel[viz.GAP + 200: viz.GAP + 200 + 2 * viz.GAP + 12, half:]
    assert count(caption_zone, lambda r, g, b: (r - g > 80) & (r - b > 80)) > 0  # the caption "no <SSEG> generated" is red


def test_long_queries_make_the_panel_taller():
    image, target = gray_image(), rect(60, 120, 10, 30, 10, 50)
    short = viz.render_panel(image, "Segment the cars.", target, None, None, "", 320)
    long = viz.render_panel(image, " ".join(["Segment the cars parked on the left side of the road"] * 6), target, None, None, "", 320)
    assert long.shape[0] > short.shape[0] and long.shape[1] == short.shape[1]
    with pytest.raises(ValueError, match="panel_width"):
        viz.render_panel(image, "x", target, None, panel_width=10)


# ----------------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------------
def test_select_indices_is_seeded_per_set_and_complete():
    names = ["semantic"] * 6 + ["conditional"] * 6 + ["attribute"] * 2
    picked = viz.select_indices(names, ["conditional", "semantic"], 3, seed=1)
    assert picked == viz.select_indices(names, ["conditional", "semantic"], 3, seed=1)
    assert len(picked) == 6 and [names[i] for i in picked] == ["conditional"] * 3 + ["semantic"] * 3
    assert picked[:3] == sorted(picked[:3]) and len(set(picked)) == 6
    assert viz.select_indices(names, ["attribute"], 5) == [12, 13]  # fewer records than requested: all of them
    assert viz.select_indices(names, ["semantic"], 3, seed=2) != viz.select_indices(names, ["semantic"], 3, seed=3)
    with pytest.raises(ValueError, match="no evaluation record"):
        viz.select_indices(names, ["unlabeled"], 2)
    with pytest.raises(ValueError, match="at least 1"):
        viz.select_indices(names, ["semantic"], 0)


# ----------------------------------------------------------------------------------------------
# Tiny model on synthetic samples
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    cfg = tiny_config()
    model, tokenizer = tiny_model_and_tokenizer(cfg)
    return cfg, model, tokenizer, model.spec


def synthetic_image(index: int, h: int = 40, w: int = 60) -> np.ndarray:
    rng = np.random.RandomState(index)
    base = rng.randint(60, 200, size=(3,), dtype=np.uint8)
    image = np.tile(base, (h, w, 1)).astype(np.uint8)
    image[::7] += 20
    return image


class SyntheticSource:
    """Two conditional and one semantic evaluation sample on random images; mimics EvalPanelSource."""

    def __init__(self, cfg: Config, spec) -> None:
        self.cfg, self.spec = cfg, spec
        self.items: List[viz.PanelInput] = []
        for k in range(3):
            image = synthetic_image(k)
            target = rect(40, 60, 5, 20, 5 + 3 * k, 30)
            anchor = rect(40, 60, 25, 38, 30, 55) if k < 2 else None
            query = "Segment the objects on the left of <REF>." if anchor is not None else "Segment all objects."
            self.items.append(viz.PanelInput(image, query, target, anchor, C.SET_CONDITIONAL if anchor is not None else C.SET_SEMANTIC,
                                             f"val/town/frame_{k}", f"rec{k}"))

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, index: int) -> Sample:
        item = self.items[index]
        sam, resize_hw = sam_preprocess(item.image, self.cfg.data.sam_image_size)
        clip = clip_preprocess(item.image, self.spec.clip_image_size, self.cfg.data.clip_preprocess)
        ref_pixels = ref_box = None
        if item.anchor is not None:
            ref_pixels = clip_preprocess(anchor_crop(item.image, item.anchor), self.spec.clip_image_size, self.cfg.data.clip_preprocess)
            ref_box = anchor_box(item.anchor, self.cfg.data.sam_image_size)
        conversation = Conversation(item.query, f"It is {C.SEG_TOKEN}.", torch.from_numpy(item.target.astype(np.uint8)), 0,
                                    item.set_name, ref_pixels, ref_box, record_id=item.record_id)
        return Sample(item.image_id, sam, clip, resize_hw, item.image.shape[:2], [conversation], "eval")

    def panel_input(self, index: int) -> viz.PanelInput:
        return self.items[index]


def test_predict_teacher_forced_gives_masks_at_the_original_size(tiny):
    cfg, model, tokenizer, spec = tiny
    source = SyntheticSource(cfg, spec)
    collate = make_collate(tokenizer, spec, cfg)
    predictions = viz.predict(model, collate, [source.sample(0), source.sample(2)], torch.device("cpu"))
    assert len(predictions) == 2
    for pred in predictions:
        assert pred.text is None and pred.mask.dtype == bool and pred.mask.shape == (40, 60)
    with pytest.raises(ValueError, match="unknown mode"):
        viz.predict(model, collate, [source.sample(0)], torch.device("cpu"), mode="beam")


def test_predict_restores_the_training_flag(tiny):
    cfg, model, tokenizer, spec = tiny
    collate = make_collate(tokenizer, spec, cfg)
    sample = SyntheticSource(cfg, spec).sample(0)
    model.train()
    viz.predict(model, collate, [sample], torch.device("cpu"))
    assert model.training
    model.eval()
    viz.predict(model, collate, [sample], torch.device("cpu"))
    assert not model.training


def test_visualize_writes_panels_and_index(tiny, tmp_path):
    cfg, model, tokenizer, spec = tiny
    source = SyntheticSource(cfg, spec)
    collate = make_collate(tokenizer, spec, cfg)
    paths = viz.visualize(model, collate, source, [0, 2, 1], tmp_path / "out", torch.device("cpu"), batch_size=2, panel_width=240)
    assert [p.name for p in paths] == [
        "000_conditional_val_town_frame_0.png", "001_semantic_val_town_frame_2.png", "002_conditional_val_town_frame_1.png"]
    for path in paths:
        image = cv2.imread(str(path))
        assert image is not None and image.shape[1] == 2 * 240 + 3 * viz.GAP and image.shape[0] > 100
    entries = json.loads((tmp_path / "out" / "index.json").read_text())
    assert [e["record_id"] for e in entries] == ["rec0", "rec2", "rec1"]
    assert all(e["mode"] == "teacher_forced" and e["prediction_available"] and 0.0 <= e["iou"] <= 1.0 for e in entries)
    assert entries[0]["query"] == "Segment the objects on the left of <REF>." and entries[1]["set"] == "semantic"
    # The target and the anchor are drawn in the left image of the conditional panel.
    panel = cv2.cvtColor(cv2.imread(str(paths[0])), cv2.COLOR_BGR2RGB)
    left = panel[viz.GAP:viz.GAP + 160, viz.GAP:viz.GAP + 240]
    assert count(left, is_red) > 0 and count(left, is_green) > 0


def test_visualize_generate_mode_handles_a_missing_mask(tiny, tmp_path):
    cfg, model, tokenizer, spec = tiny
    source = SyntheticSource(cfg, spec)
    collate = make_collate(tokenizer, spec, cfg, generation=True)
    paths = viz.visualize(model, collate, source, [0, 1], tmp_path, torch.device("cpu"), mode="generate", max_new_tokens=4,
                          panel_width=200)
    assert len(paths) == 2
    entries = json.loads((tmp_path / "index.json").read_text())
    for entry in entries:
        assert entry["mode"] == "generate" and isinstance(entry["generated_text"], str)
        assert entry["prediction_available"] in (True, False)
        if not entry["prediction_available"]:
            assert entry["iou"] == 0.0  # no <SSEG>, no mask: counts as an empty prediction


def test_visualize_rejects_bad_selections(tiny, tmp_path):
    cfg, model, tokenizer, spec = tiny
    source = SyntheticSource(cfg, spec)
    collate = make_collate(tokenizer, spec, cfg)
    with pytest.raises(ValueError, match="no samples"):
        viz.visualize(model, collate, source, [], tmp_path, torch.device("cpu"))
    with pytest.raises(IndexError, match="outside"):
        viz.visualize(model, collate, source, [7], tmp_path, torch.device("cpu"))
    with pytest.raises(ValueError, match="batch_size"):
        viz.visualize(model, collate, source, [0], tmp_path, torch.device("cpu"), batch_size=0)


def test_write_panels_checks_lengths(tmp_path):
    item = viz.PanelInput(gray_image(), "q", rect(60, 120, 0, 10, 0, 10), None, "semantic", "a/b", None)
    with pytest.raises(ValueError, match="predictions"):
        viz.write_panels([item], [], tmp_path)
    paths = viz.write_panels([item], [viz.Prediction(None)], tmp_path)
    assert paths[0].name == "000_semantic_a_b.png"


# ----------------------------------------------------------------------------------------------
# EvalDataset records on a synthetic Cityscapes folder
# ----------------------------------------------------------------------------------------------
def make_cityscapes(root: Path) -> str:
    """One validation image with two cars (labelId 26 -> trainId 13) and a road; returns its image id."""
    image_id = "val/town/town_000000_000019"
    (root / "leftImg8bit" / "val" / "town").mkdir(parents=True)
    (root / "gtFine" / "val" / "town").mkdir(parents=True)
    rgb = synthetic_image(5, 48, 96)
    cv2.imwrite(str(root / "leftImg8bit" / f"{image_id}_leftImg8bit.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    label = np.full((48, 96), 7, np.uint8)  # road
    label[10:30, 8:30] = 26  # car 1
    label[10:30, 60:90] = 26  # car 2
    cv2.imwrite(str(root / "gtFine" / f"{image_id}_gtFine_labelIds.png"), label)
    return image_id


def test_eval_panel_source_reads_records_targets_and_anchors(tmp_path):
    image_id = make_cityscapes(tmp_path / "cityscapes")
    cfg = tiny_config()
    cfg.paths.cityscapes_root = str(tmp_path / "cityscapes")
    cfg.paths.data_dir = str(tmp_path / "data")
    cfg.model.reference.mode = "tokens"
    car = ObjectRef(13, "car", ["13_1"])
    other = ObjectRef(13, "car", ["13_2"])
    conditional = InstructionRecord("c0", "cityscapes", "val", image_id, C.SET_CONDITIONAL, f"Segment the car on the left of {C.REF_TOKEN}.",
                                    car, anchor=other, relation="to the left of")
    semantic = InstructionRecord("s0", "cityscapes", "val", image_id, C.SET_SEMANTIC, "Segment the road.", ObjectRef(0, "road", []))
    write_jsonl(cfg.instructions_path("val", C.SET_CONDITIONAL), [conditional])
    write_jsonl(cfg.instructions_path("val", C.SET_SEMANTIC), [semantic])

    model, tokenizer = tiny_model_and_tokenizer(cfg)
    dataset = EvalDataset(cfg, model.spec, [C.SET_CONDITIONAL, C.SET_SEMANTIC], split="val")
    source = viz.EvalPanelSource(dataset)
    assert len(source) == 2 and source.set_names == [C.SET_CONDITIONAL, C.SET_SEMANTIC]

    item = source.panel_input(0)
    assert item.image.shape == (48, 96, 3) and item.record_id == "c0" and item.set_name == C.SET_CONDITIONAL
    assert item.target.sum() == 20 * 22 and item.anchor.sum() == 20 * 30 and not (item.target & item.anchor).any()
    assert source.panel_input(1).anchor is None and source.panel_input(1).target.sum() == 48 * 96 - (20 * 22 + 20 * 30)

    collate = make_collate(tokenizer, model.spec, cfg)
    paths = viz.visualize(model, collate, source, [0, 1], tmp_path / "panels", torch.device("cpu"), panel_width=240)
    assert len(paths) == 2 and all(p.is_file() for p in paths)
    entries = json.loads((tmp_path / "panels" / "index.json").read_text())
    assert [e["record_id"] for e in entries] == ["c0", "s0"]


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------
def test_main_runs_end_to_end_on_the_tiny_model(tmp_path):
    image_id = make_cityscapes(tmp_path / "cityscapes")
    config = tmp_path / "tiny.yaml"
    config.write_text(
        f"paths:\n  cityscapes_root: {tmp_path / 'cityscapes'}\n  data_dir: {tmp_path / 'data'}\n  output_dir: {tmp_path / 'outputs'}\n"
        "model:\n  tiny: true\n  precision: fp32\n"
        "data:\n  sam_image_size: 64\n  num_workers: 0\n"
        "eval:\n  sets: [conditional, semantic]\n"
    )
    cfg = tiny_config()
    cfg.paths.data_dir = str(tmp_path / "data")
    write_jsonl(cfg.instructions_path("val", C.SET_CONDITIONAL), [InstructionRecord(
        "c0", "cityscapes", "val", image_id, C.SET_CONDITIONAL, f"Segment the car on the left of {C.REF_TOKEN}.",
        ObjectRef(13, "car", ["13_1"]), anchor=ObjectRef(13, "car", ["13_2"]), relation="to the left of")])
    write_jsonl(cfg.instructions_path("val", C.SET_SEMANTIC), [InstructionRecord(
        "s0", "cityscapes", "val", image_id, C.SET_SEMANTIC, "Segment the road.", ObjectRef(0, "road", []))])
    out = tmp_path / "panels"
    assert viz.main(["--config", str(config), "--no-checkpoint", "--num-samples", "2", "--panel-width", "200", "--out-dir", str(out)]) == 0
    entries = json.loads((out / "index.json").read_text())
    assert [e["set"] for e in entries] == ["conditional", "semantic"] and all((out / e["file"]).is_file() for e in entries)
    # An explicit record index and a missing checkpoint.
    out2 = tmp_path / "panels2"
    assert viz.main(["--config", str(config), "--no-checkpoint", "--indices", "1", "--out-dir", str(out2)]) == 0
    assert [e["record_id"] for e in json.loads((out2 / "index.json").read_text())] == ["s0"]
    with pytest.raises(FileNotFoundError, match="no finished stage"):
        viz.main(["--config", str(config), "--out-dir", str(tmp_path / "none")])


def test_parse_args_defaults_and_choices():
    args = viz.parse_args(["--config", "configs/cityscapes/1_16.yaml", "--sets", "conditional", "--num-samples", "3", "data.partition=1_16"])
    assert args.sets == ["conditional"] and args.num_samples == 3 and args.overrides == ["data.partition=1_16"]
    assert args.split == "val" and args.mode is None and args.panel_width == 640
    with pytest.raises(SystemExit):
        viz.parse_args(["--mode", "beam"])
    with pytest.raises(SystemExit):
        viz.parse_args(["--sets", "unlabeled"])


def test_default_checkpoint_needs_a_finished_stage(tmp_path):
    cfg = tiny_config()
    cfg.paths.output_dir = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="no finished stage"):
        viz.default_checkpoint(cfg)
    final = tmp_path / cfg.train.run_name / "stage2" / "final.pt"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"x")
    assert viz.default_checkpoint(cfg) == final


# ----------------------------------------------------------------------------------------------
# Free-form queries (Fig. 1, Fig. 6, Fig. 7)
# ----------------------------------------------------------------------------------------------
def tiny_cityscapes_config(tmp_path: Path) -> Config:
    cfg = tiny_config()
    cfg.paths.cityscapes_root = str(tmp_path / "cityscapes")
    cfg.paths.data_dir = str(tmp_path / "data")
    cfg.model.reference.mode = "tokens"
    return cfg


def test_demo_queries_are_the_paper_texts():
    assert viz.DEMO_QUERIES["fig1"] == ("Segment the rightmost four wheeler to transport people",
                                        "Segment the surface that a car can drive on")
    assert any("sitting on and riding the bicycle" in q for q in viz.DEMO_QUERIES["fig7"])
    assert any("on the road and behind a truck" in q for q in viz.DEMO_QUERIES["fig7"])


def test_panel_without_target_draws_no_red_ground_truth():
    image = gray_image()
    panel = viz.render_panel(image, "Segment the road", None, None, None, panel_width=160)
    left_image = panel[viz.GAP:viz.GAP + 80, viz.GAP:viz.GAP + 160]  # the caption of a missing prediction is drawn in red
    assert left_image.shape[:2] == (80, 160) and count(left_image, is_red) == 0 and count(left_image, is_green) == 0
    assert viz.mask_iou(None, rect(60, 120, 0, 10, 0, 10)) == 0.0
    assert viz.mask_iou(rect(60, 120, 0, 10, 0, 10), None) is None


def test_query_source_builds_samples_with_and_without_anchor(tmp_path):
    image_id = make_cityscapes(tmp_path / "cityscapes")
    cfg = tiny_cityscapes_config(tmp_path)
    model, tokenizer = tiny_model_and_tokenizer(cfg)
    source = viz.QueryPanelSource(cfg, model.spec, "val", [
        viz.FreeQuery("Segment the surface that a car can drive on"),
        viz.FreeQuery(f"Segment the car on the left of {C.REF_TOKEN}", image_id, anchor="13_2", target="13_1"),
    ])
    assert len(source) == 2 and source.queries[0].image_id == image_id  # the first image of the split is the default
    plain, conditional = source.sample(0).conversations[0], source.sample(1).conversations[0]
    assert plain.ref_pixels is None and plain.ref_box is None and int(plain.mask.sum()) == 0
    assert conditional.ref_pixels is not None and conditional.ref_box is not None and conditional.set_name == "freeform"
    item = source.panel_input(1)
    assert item.target.sum() == 20 * 22 and item.anchor.sum() == 20 * 30
    assert source.panel_input(0).target is None and source.panel_input(0).anchor is None


def test_query_source_rejects_bad_queries(tmp_path):
    make_cityscapes(tmp_path / "cityscapes")
    cfg = tiny_cityscapes_config(tmp_path)
    model, _ = tiny_model_and_tokenizer(cfg)

    def build(*queries):
        return viz.QueryPanelSource(cfg, model.spec, "val", list(queries))

    with pytest.raises(ValueError, match="no query"):
        build()
    with pytest.raises(ValueError, match="gives no anchor"):
        build(viz.FreeQuery(f"Segment the car on the left of {C.REF_TOKEN}"))
    with pytest.raises(ValueError, match="no <REF>"):
        build(viz.FreeQuery("Segment the car", anchor="13_1"))
    with pytest.raises(ValueError, match="at most one"):
        build(viz.FreeQuery(f"{C.REF_TOKEN} next to {C.REF_TOKEN}", anchor="13_1"))
    with pytest.raises(ValueError, match="must not contain"):
        build(viz.FreeQuery(f"Segment {C.SEG_TOKEN}"))
    with pytest.raises(ValueError, match="bad region id"):
        build(viz.FreeQuery("Segment the car", target="car"))
    with pytest.raises(ValueError, match="class ids are"):
        build(viz.FreeQuery("Segment the car", target="99_1"))
    with pytest.raises(ValueError, match="empty"):
        build(viz.FreeQuery("  "))


def test_queries_file_and_command_line_collection(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"query": "Segment the road", "image_id": "a"}, {"query": f"Segment the car left of {C.REF_TOKEN}", "anchor": "13_2"}]))
    loaded = viz.load_queries_file(path)
    assert [q.image_id for q in loaded] == ["a", None] and loaded[1].anchor == "13_2"
    path.write_text(json.dumps([{"query": "x", "colour": "red"}]))
    with pytest.raises(ValueError, match="only the keys"):
        viz.load_queries_file(path)
    path.write_text("[]")
    with pytest.raises(ValueError, match="non-empty"):
        viz.load_queries_file(path)

    args = viz.parse_args(["--demo", "fig1", "--demo", "fig7", "--query", "Segment the road", "--image-id", "val/x"])
    queries = viz.collect_queries(args)
    assert [q.query for q in queries] == list(viz.FIG1_QUERIES + viz.FIG7_QUERIES) + ["Segment the road"]
    assert all(q.image_id == "val/x" for q in queries)
    assert viz.collect_queries(viz.parse_args([])) == []
    with pytest.raises(ValueError, match="need --query"):
        viz.collect_queries(viz.parse_args(["--anchor", "13_1"]))


def test_main_draws_the_fig1_probes_and_a_ref_query(tmp_path):
    image_id = make_cityscapes(tmp_path / "cityscapes")
    config = tmp_path / "tiny.yaml"
    config.write_text(
        f"paths:\n  cityscapes_root: {tmp_path / 'cityscapes'}\n  data_dir: {tmp_path / 'data'}\n  output_dir: {tmp_path / 'outputs'}\n"
        "model:\n  tiny: true\n  precision: fp32\n  reference:\n    mode: tokens\n"
        "data:\n  sam_image_size: 64\n  num_workers: 0\n"
    )
    out = tmp_path / "demo"
    argv = ["--config", str(config), "--no-checkpoint", "--demo", "fig1", "--panel-width", "200", "--out-dir", str(out)]
    assert viz.main(argv) == 0
    entries = json.loads((out / "index.json").read_text())
    assert [e["query"] for e in entries] == list(viz.FIG1_QUERIES)
    assert all(e["set"] == "freeform" and e["iou"] is None and (out / e["file"]).is_file() for e in entries)

    out2 = tmp_path / "ref"
    argv = ["--config", str(config), "--no-checkpoint", "--image-id", image_id, "--anchor", "13_2", "--target", "13_1",
            "--query", f"Segment the individual who is riding {C.REF_TOKEN}", "--mode", "generate", "--max-new-tokens", "4",
            "--panel-width", "200", "--out-dir", str(out2)]
    assert viz.main(argv) == 0
    entry = json.loads((out2 / "index.json").read_text())[0]
    assert entry["image_id"] == image_id and entry["iou"] is not None and entry["mode"] == "generate"
    with pytest.raises(ValueError, match="cannot be combined"):
        viz.main(["--config", str(config), "--no-checkpoint", "--demo", "fig1", "--indices", "0"])
