"""Tests of cora.data.reasonseg: parquet copies, LISA folder layout, prompts and the long/short flag (Table 3)."""
import io
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from cora import constants as C
from cora.config import Config
from cora.data.collate import collate
from cora.data.conversation import encode
from cora.data.reasonseg import (
    REASONSEG_ANSWER, ReasonSegDataset, load_flag_manifest, rasterise_shapes, reasonseg_query,
)
from tests.test_conversation import get_tokenizer

# JSON with the long/short flag per ReasonSeg val image (see paths.reasonseg_manifest); the test that needs it is skipped
# when CORA_REASONSEG_MANIFEST is not set.
REAL_MANIFEST = Path(os.environ["CORA_REASONSEG_MANIFEST"]) if os.environ.get("CORA_REASONSEG_MANIFEST") else None
SIZES = [(30, 50), (40, 24), (20, 20), (36, 60), (28, 28)]
TEXTS = ["the cesspit", "In an orchestra, musicians play different instruments. What object is the biggest?",
         "a red car", "the thing that holds water", "dog"]


def make_rows(count=5):
    rows = []
    for i in range(count):
        h, w = SIZES[i]
        rng = np.random.default_rng(i)
        image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")  # lossless, so pixels can be compared
        mask = np.zeros((h, w), dtype=bool)
        if i != 3:  # row 3 has an empty target
            mask[2 + i : 8 + i, 3:10] = True
        rows.append(dict(image=image, png=buffer.getvalue(), mask=mask, text=TEXTS[i], image_id=f"img{i}"))
    return rows


def write_parquet(path: Path, rows, row_group_size=2):
    table = pa.table({
        "image": pa.array([{"bytes": r["png"], "path": r["image_id"] + ".jpg"} for r in rows],
                          type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "text": [r["text"] for r in rows],
        "mask": pa.array([r["mask"].tolist() for r in rows], type=pa.list_(pa.list_(pa.bool_()))),
        "image_id": [r["image_id"] for r in rows],
        "ann_id": [r["image_id"] for r in rows],
        "img_height": [r["mask"].shape[0] for r in rows],
        "img_width": [r["mask"].shape[1] for r in rows],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size)


@pytest.fixture()
def parquet_root(tmp_path):
    rows = make_rows()
    write_parquet(tmp_path / "reasonseg_test" / "data" / "test-00000-of-00002.parquet", rows[:3])
    write_parquet(tmp_path / "reasonseg_test" / "data" / "test-00001-of-00002.parquet", rows[3:])
    return tmp_path, rows


def make_cfg(root, manifest=None):
    cfg = Config()
    cfg.paths.reasonseg_root = str(root)
    cfg.paths.reasonseg_manifest = None if manifest is None else str(manifest)
    cfg.data.sam_image_size = 64
    return cfg


def spec():
    return get_tokenizer(4, 28)[1]


# ----------------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------------
def test_prompts_follow_lisa_validation():
    # LISA's ValDataset: "<image>\n {text} ..." and "<image>\n What is {text} in this image? ..." (space after the newline)
    assert reasonseg_query("  the cesspit ", False) == " What is the cesspit in this image? Please output segmentation mask."
    assert reasonseg_query("What object holds water?", True) == " What object holds water? Please output segmentation mask."
    assert reasonseg_query("What object holds water?", None) == " What object holds water? Please output segmentation mask."
    assert REASONSEG_ANSWER == "<SSEG>."
    with pytest.raises(ValueError, match="empty"):
        reasonseg_query("  ", True)


def test_prompt_tokens_match_lisa_after_the_image_block():
    """LISA's text after "<image>\\n" starts with a space, so the first word is a '▁'-prefixed piece."""
    tok, sp = get_tokenizer(4, 28)
    for is_long, word in ((True, "▁What"), (False, "▁What")):
        ids, labels = encode(tok, sp, reasonseg_query("What is this?" if is_long else "cake", is_long), REASONSEG_ANSWER, "SYS", 64)
        pieces = tok.convert_ids_to_tokens(ids.tolist())
        at = max(i for i, t in enumerate(pieces) if t == "<0x0A>")  # the newline after the image block
        assert pieces[at + 1] == word
    # the reference: LISA's text without the space gives the piece without "▁"
    ids, _ = encode(tok, sp, "What is this?", REASONSEG_ANSWER, "SYS", 64)
    pieces = tok.convert_ids_to_tokens(ids.tolist())
    assert pieces[max(i for i, t in enumerate(pieces) if t == "<0x0A>") + 1] == "What"


# ----------------------------------------------------------------------------------------------
# Parquet copies
# ----------------------------------------------------------------------------------------------
def test_parquet_dataset_reads_rows_lazily_and_in_any_order(parquet_root):
    root, rows = parquet_root
    ds = ReasonSegDataset(make_cfg(root), spec(), "test")
    assert len(ds) == 5 and ds.image_ids == [r["image_id"] for r in rows]
    for index in (0, 1, 2, 3, 4, 4, 1, 0, 3, 2):  # forward, repeated and backward access
        sample = ds[index]
        row = rows[index]
        h, w = row["mask"].shape
        assert sample.kind == "eval" and sample.image_id == row["image_id"] and sample.orig_hw == (h, w)
        assert sample.image_sam.shape == (3, 64, 64) and sample.image_clip.shape == (3, 28, 28)
        conv = sample.conversations[0]
        assert conv.set_name == "reasonseg" and conv.class_id == -1 and conv.answer == "<SSEG>." and conv.record_id == row["image_id"]
        assert conv.mask.dtype == torch.uint8 and conv.mask.shape == (h, w)
        assert np.array_equal(conv.mask.numpy(), row["mask"].astype(np.uint8))
        assert row["text"].strip() in conv.query and conv.is_long_query is None
    with pytest.raises(IndexError):
        ds[5]


def test_parquet_image_with_an_exif_orientation_is_rotated_like_its_mask(tmp_path):
    """Some ReasonSeg photos carry an EXIF orientation tag; their masks are drawn in the rotated frame."""
    stored = np.zeros((20, 30, 3), dtype=np.uint8)  # as stored: 20 rows, 30 columns
    stored[:, :10] = 255  # a white band on the left, which the rotation moves to the top
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"
    buffer = io.BytesIO()
    Image.fromarray(stored).save(buffer, format="JPEG", quality=95, exif=exif.tobytes())
    mask = np.zeros((30, 20), dtype=bool)  # the displayed frame: 30 rows, 20 columns
    mask[:10] = True
    row = dict(image=stored, png=buffer.getvalue(), mask=mask, text="the white band", image_id="rotated")
    write_parquet(tmp_path / "reasonseg_test" / "data" / "test-00000-of-00001.parquet", [row])
    ds = ReasonSegDataset(make_cfg(tmp_path), spec(), "test")
    sample = ds[0]
    assert sample.orig_hw == (30, 20) and sample.conversations[0].mask.shape == (30, 20)
    image, target = ds._rows.read(0)
    assert image.shape == (30, 20, 3)
    assert image[target].mean() > 200 and image[~target].mean() < 50  # the mask covers the white band


def test_flag_join_selects_the_prompt(parquet_root, tmp_path):
    root, rows = parquet_root
    manifest = tmp_path / "manifest.json"
    flags = [False, True, False, False, False]
    manifest.write_text(json.dumps({"test": {"images": [{"image_id": r["image_id"], "is_sentence": f, "n_phrasings": 1}
                                                          for r, f in zip(rows, flags)]}}))
    ds = ReasonSegDataset(make_cfg(root, manifest), spec(), "test")
    assert [ds.is_long_query(i) for i in range(5)] == flags
    short, long_ = ds[0].conversations[0], ds[1].conversations[0]
    assert short.is_long_query is False and short.query == " What is the cesspit in this image? Please output segmentation mask."
    assert long_.is_long_query is True and long_.query == " " + TEXTS[1] + " Please output segmentation mask."
    # the batch carries the flag, so evaluation can report the long-query subset
    tok, sp = get_tokenizer(4, 28)
    batch = collate([ds[0], ds[1]], tok, sp, make_cfg(root))
    assert batch.is_long_query == [False, True] and batch.set_names == ["reasonseg"] * 2
    assert batch.answers == ["<SSEG>."] * 2
    assert not hasattr(ds, "guess")  # the flag is never inferred from the text


def test_manifest_forms_and_errors(parquet_root, tmp_path):
    root, rows = parquet_root
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"images": [{"image_id": "a", "is_sentence": True}]}))
    assert load_flag_manifest(str(flat), "test") == {"a": True}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"test": {"images": [{"image_id": "a"}]}}))
    with pytest.raises(ValueError, match="is_sentence"):
        load_flag_manifest(str(bad), "test")
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"test": {"images": [{"image_id": "img0", "is_sentence": False}]}}))
    with pytest.raises(ValueError, match="no long/short flag for 4 images"):
        ReasonSegDataset(make_cfg(root, partial), spec(), "test")


def test_parquet_dataset_is_picklable_after_reading(parquet_root):
    root, _ = parquet_root
    ds = ReasonSegDataset(make_cfg(root), spec(), "test")
    ds[1]
    clone = pickle.loads(pickle.dumps(ds))
    assert np.array_equal(clone[1].conversations[0].mask.numpy(), ds[1].conversations[0].mask.numpy())


def test_dataset_errors(tmp_path):
    with pytest.raises(ValueError, match="split"):
        ReasonSegDataset(make_cfg(tmp_path), spec(), "dev")
    with pytest.raises(FileNotFoundError, match="no ReasonSeg test data"):
        ReasonSegDataset(make_cfg(tmp_path), spec(), "test")


# ----------------------------------------------------------------------------------------------
# LISA folder layout
# ----------------------------------------------------------------------------------------------
def test_rasterise_shapes_matches_lisa_rules():
    big = {"label": "target", "points": [[0, 0], [30, 0], [30, 30], [0, 30]]}
    small_ignore = {"label": "Ignore", "points": [[10, 10], [14, 10], [14, 14], [10, 14]]}
    flag = {"label": "flag", "points": [[0, 0], [39, 0], [39, 39], [0, 39]]}
    mask = rasterise_shapes([small_ignore, flag, big], 40, 40)
    assert mask.dtype == np.uint8 and set(np.unique(mask).tolist()) == {0, 1, 255}
    assert mask[12, 12] == 255 and mask[5, 5] == 1 and mask[35, 35] == 0  # the small ignore region is drawn last
    assert rasterise_shapes([], 5, 6).shape == (5, 6)


def test_folder_layout_uses_first_text_and_json_flag(tmp_path):
    folder = tmp_path / "ReasonSeg" / "val"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, is_sentence, texts in (("a", True, ["long one", "second"]), ("b", False, ["the dog"])):
        cv2.imwrite(str(folder / f"{name}.jpg"), rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))
        shapes = [{"label": "target", "points": [[5, 5], [30, 5], [30, 25], [5, 25]]},
                  {"label": "ignore", "points": [[20, 10], [25, 10], [25, 15], [20, 15]]}]
        (folder / f"{name}.json").write_text(json.dumps({"text": texts, "is_sentence": is_sentence, "shapes": shapes}))
    ds = ReasonSegDataset(make_cfg(tmp_path), spec(), "val")
    assert ds.image_ids == ["a", "b"] and [ds.is_long_query(i) for i in range(2)] == [True, False]
    a, b = ds[0].conversations[0], ds[1].conversations[0]
    assert a.query == " long one Please output segmentation mask." and a.is_long_query is True
    assert b.query == " What is the dog in this image? Please output segmentation mask."
    assert set(np.unique(a.mask.numpy()).tolist()) == {0, 1, 255} and ds[0].orig_hw == (40, 60)
    (folder / "b.json").unlink()
    with pytest.raises(FileNotFoundError, match="no .json"):
        ReasonSegDataset(make_cfg(tmp_path), spec(), "val")


# ----------------------------------------------------------------------------------------------
# Real parquet copies
# ----------------------------------------------------------------------------------------------
@pytest.mark.data
def test_real_parquet_sample(reasonseg_root):
    cfg = make_cfg(reasonseg_root)
    cfg.data.sam_image_size = 1024
    ds = ReasonSegDataset(cfg, get_tokenizer(576, 336)[1], "val")
    assert len(ds) == 200
    sample = ds[0]
    h, w = sample.orig_hw
    conv = sample.conversations[0]
    assert (h, w) == (1712, 2288) and conv.mask.shape == (h, w) and conv.mask.dtype == torch.uint8
    assert set(conv.mask.unique().tolist()) <= {0, 1} and conv.mask.sum() > 0
    assert sample.image_sam.shape == (3, 1024, 1024) and sample.image_clip.shape == (3, 336, 336)
    assert sample.resize_hw == (766, 1024)
    assert conv.query.endswith("Please output segmentation mask.") and conv.is_long_query is None
    assert ds[1].orig_hw != sample.orig_hw  # a later row of the same row group is reached by streaming
    assert ds[150].conversations[0].mask.shape == ds[150].orig_hw  # a row of the second row group


@pytest.mark.data
@pytest.mark.skipif(REAL_MANIFEST is None or not REAL_MANIFEST.is_file(), reason="set CORA_REASONSEG_MANIFEST to the manifest file")
def test_real_manifest_join_counts(reasonseg_root):
    cfg = make_cfg(reasonseg_root, REAL_MANIFEST)
    ds = ReasonSegDataset(cfg, get_tokenizer(4, 28)[1], "val")
    flags = [ds.is_long_query(i) for i in range(len(ds))]
    assert (sum(flags), len(flags) - sum(flags)) == (113, 87)
    short_index = flags.index(False)
    assert ds[short_index].conversations[0].query.startswith(" What is ")


# ----------------------------------------------------------------------------------------------
# The train split as training data (data.reasonseg_train_share)
# ----------------------------------------------------------------------------------------------
def prepare_train(tmp_path, flags=(False, True, False, None, False), broken=()):
    """Synthetic train parquet -> tools/prepare_reasonseg.py; returns (cfg, rows, printed JSON)."""
    import contextlib
    import importlib.util

    rows = make_rows()
    for i in broken:  # a mask on a canvas of another shape, as the rotated images of the real train copy have
        rows[i]["mask"] = np.zeros((rows[i]["mask"].shape[1] + 3, rows[i]["mask"].shape[0]), dtype=bool)
    root = tmp_path / "rs"
    write_parquet(root / "reasonseg_train" / "data" / "train-00000-of-00001.parquet", rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": {"images": [{"image_id": r["image_id"], "is_sentence": bool(f)}
                                                           for r, f in zip(rows, flags) if f is not None]}}))
    spec_ = importlib.util.spec_from_file_location("tool_prepare_reasonseg", Path(__file__).resolve().parents[1] / "tools" / "prepare_reasonseg.py")
    tool = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(tool)
    args = ["--set", f"paths.reasonseg_root={root}", "--set", f"paths.data_dir={tmp_path / 'data'}", "--set", "data.sam_image_size=64",
            "--split", "train"]
    has_all_flags = all(f is not None for f in flags)
    if has_all_flags:
        args += ["--set", f"paths.reasonseg_manifest={manifest}"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert tool.main(args) == 0
    cfg = make_cfg(root, manifest if has_all_flags else None)
    cfg.paths.data_dir = str(tmp_path / "data")
    return cfg, rows, json.loads(out.getvalue()), tool, args


def test_prepared_train_split_gives_labeled_samples_in_lisas_wording(tmp_path):
    import random

    from cora.data import templates
    from cora.data.reasonseg import ReasonSegTrainSet

    cfg, rows, printed, tool, args = prepare_train(tmp_path, flags=(False, True, False, False, False))
    folder = cfg.reasonseg_train_dir
    assert printed["images"] == 5 and printed["long_queries"] == 1 and printed["short_queries"] == 4 and printed["skipped_mask_in_another_orientation"] == []
    assert sorted(p.name for p in folder.iterdir()) == sorted([f"img{i}.{e}" for i in range(5) for e in ("jpg", "png")] + ["index.json", "skipped.json"])
    assert np.array_equal(cv2.imread(str(folder / "img0.png"), cv2.IMREAD_UNCHANGED), rows[0]["mask"].astype(np.uint8))
    with pytest.raises(SystemExit, match="already exists"):
        tool.main(args)
    train = ReasonSegTrainSet(cfg, spec())
    assert len(train) == 5
    seen = {}
    for seed in range(60):
        sample = train.sample(random.Random(seed))
        (conversation,) = sample.conversations
        assert sample.kind == "labeled" and sample.image_id.startswith("reasonseg/") and conversation.set_name == "reasonseg"
        assert conversation.class_id == -1 and conversation.answer in templates.ANSWER_LIST and conversation.answer.count(C.SEG_TOKEN) == 1
        index = int(conversation.record_id[3:])
        assert sample.orig_hw == SIZES[index] and conversation.mask.shape == SIZES[index]
        assert torch.equal(conversation.mask.bool(), torch.from_numpy(rows[index]["mask"]))
        text = TEXTS[index]
        if index == 1:  # a sentence: LISA's long templates around the text as written
            assert conversation.query in [t.format(sent=text) for t in templates.LONG_QUESTION_LIST] and conversation.is_long_query is True
        else:  # a phrase: LISA's short templates around the lower-cased text
            assert conversation.query in [t.format(class_name=text.lower()) for t in templates.SHORT_QUESTION_LIST]
        seen[index] = seen.get(index, 0) + 1
        ids, labels = encode(get_tokenizer(4, 28)[0], spec(), conversation.query, conversation.answer, Config().model.system_prompt, 384)
        assert (ids == spec().seg_token_id).sum() == 1
    assert set(seen) == {0, 1, 2, 3, 4}
    # one draw is reproducible from its generator
    a, b = train.sample(random.Random(7)), train.sample(random.Random(7))
    assert a.image_id == b.image_id and a.conversations[0].query == b.conversations[0].query and torch.equal(a.image_sam, b.image_sam)


def test_train_subset_is_fixed_and_nested():
    from cora.data.reasonseg import train_subset

    ids = [f"img{i:03d}" for i in range(232)]
    sizes = [8, 15, 29, 58]  # 1/30, 1/16, 1/8, 1/4 of the 232 images, rounded up
    subsets = [train_subset(ids, n) for n in sizes]
    assert [len(s) for s in subsets] == sizes and all(len(set(s)) == len(s) for s in subsets)
    for small, large in zip(subsets, subsets[1:]):
        assert set(small) < set(large), "a smaller fraction is contained in the larger one"
    assert train_subset(list(reversed(ids)), 15) == subsets[1], "the choice does not depend on the order of the list"
    assert sorted(train_subset(ids, 232)) == ids
    assert subsets[0] != sorted(ids)[:8], "not simply the first images by name"
    for bad in (0, 233):
        with pytest.raises(ValueError, match="data.reasonseg_train_images"):
            train_subset(ids, bad)


def test_reasonseg_train_images_limits_the_draws_to_the_subset(tmp_path):
    import random

    from cora.config import validate
    from cora.data.reasonseg import ReasonSegTrainSet, train_subset

    cfg, rows, _, _, _ = prepare_train(tmp_path, flags=(False, True, False, False, False))
    full = ReasonSegTrainSet(cfg, spec())
    assert len(full) == 5 and full.available == 5
    cfg.data.reasonseg_train_share = 0.25
    cfg.data.reasonseg_train_images = 2
    validate(cfg)
    part = ReasonSegTrainSet(cfg, spec())
    chosen = set(train_subset([r["image_id"] for r in rows], 2))
    assert len(part) == 2 and part.available == 5 and {e["image_id"] for e in part.entries} == chosen
    drawn = {part.sample(random.Random(seed)).conversations[0].record_id for seed in range(40)}
    assert drawn == chosen, "every draw comes from the subset, and both images are drawn"
    # the subset is the same for another training seed, and a larger one contains it
    cfg.train.seed = 5
    assert {e["image_id"] for e in ReasonSegTrainSet(cfg, spec()).entries} == chosen
    cfg.data.reasonseg_train_images = 4
    assert chosen < {e["image_id"] for e in ReasonSegTrainSet(cfg, spec()).entries}
    cfg.data.reasonseg_train_images = 6
    with pytest.raises(ValueError, match="has 5 images"):
        ReasonSegTrainSet(cfg, spec())
    # settings that cannot work are refused when the configuration is read
    cfg.data.reasonseg_train_images = 0
    with pytest.raises(ValueError, match="at least 1"):
        validate(cfg)
    cfg.data.reasonseg_train_images = 2
    cfg.data.reasonseg_train_share = 0.0
    with pytest.raises(ValueError, match="no ReasonSeg sample would be drawn"):
        validate(cfg)


def test_prepare_leaves_out_masks_drawn_in_another_orientation_and_unknown_flags_use_the_long_wording(tmp_path):
    import random

    from cora.data import templates
    from cora.data.reasonseg import ReasonSegTrainSet

    cfg, rows, printed, _, _ = prepare_train(tmp_path, flags=(None,) * 5, broken=(2,))
    assert printed["images"] == 4 and printed["skipped_mask_in_another_orientation"] == ["img2"] and printed["flag_unknown"] == 4
    assert json.loads((cfg.reasonseg_train_dir / "skipped.json").read_text()) == ["img2"]
    train = ReasonSegTrainSet(cfg, spec())
    queries = {train.sample(random.Random(s)).conversations[0].query for s in range(30)}
    assert all(any(q == t.format(sent=text) for t in templates.LONG_QUESTION_LIST for text in TEXTS) for q in queries)
    # without the prepared folder the error names the command
    cfg.paths.data_dir = str(tmp_path / "elsewhere")
    with pytest.raises(FileNotFoundError, match="prepare_reasonseg.py --split train"):
        ReasonSegTrainSet(cfg, spec())
