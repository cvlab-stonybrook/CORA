"""Tests of cora.data.adapters / cora.data.cityscapes (paper Sec. 4.1-4.2) and tools/prepare_cityscapes.py."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from cora import constants as C
from cora.config import Config
from cora.data.adapters import DatasetAdapter, atomic_write, get_adapter, read_id_list
from cora.data.cityscapes import CityscapesAdapter, labelid_lut, split_line_to_id

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "assets" / "splits" / "cityscapes"
COUNTS = {"1_30": (100, 2875), "1_16": (186, 2789), "1_8": (372, 2603), "1_4": (744, 2231)}
SHA256 = {  # the lists are copied byte for byte from the UniMatch / SemiVL repositories
    "1_16/labeled.txt": "64195914dce1231133c7dc611fece213bc3a87fe4e0b73fbd96add7dfa8a2f46",
    "1_16/unlabeled.txt": "7f18e2aefb5e88598910e291a935117daaeef51c6828edeb62f8a6c882be0a79",
    "1_30/labeled.txt": "b11041fef70476fb5564af0ef006a8013622da9cf49c7faf4182f3e26eb32742",
    "1_30/unlabeled.txt": "1b8a9b7bb4f9f6f06a3dffe766efe8b10a14a0137637af1738bbb417a5c85dbe",
    "1_4/labeled.txt": "8edd24ae817eed7b49d0a06c95b7bfaf769002c98387e4edeab5e221ab7791c3",
    "1_4/unlabeled.txt": "6fc53258ed2e40f72f4a22b6e20d60a789168fdf6b860033f826e8bee3900e9b",
    "1_8/labeled.txt": "97291dfc509a8c90ee94e1c5fd1f7668c21205b995f53b963ac133f0ed96ee8e",
    "1_8/unlabeled.txt": "5d7d33e39a8bda908e12d7f33ecb3299f93a8596fbe3c5e0cc1f2d3f5d358459",
    "val.txt": "cf5aa35c0e4446bc684b61b6be88e22299ef7b1851852b53fc6a34794f161544",
}


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------------------------
# Split lists (no dataset needed)
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("relative", sorted(SHA256))
def test_split_files_are_the_original_bytes(relative):
    assert hashlib.sha256((SPLITS / relative).read_bytes()).hexdigest() == SHA256[relative]


def test_split_counts_disjoint_and_cover_training_set():
    adapter = CityscapesAdapter(Path("/nonexistent"))
    train_ids = set()
    for partition, (n_labeled, n_unlabeled) in COUNTS.items():
        labeled, unlabeled = adapter.partition(partition, "labeled"), adapter.partition(partition, "unlabeled")
        assert (len(labeled), len(unlabeled)) == (n_labeled, n_unlabeled)
        assert len(set(labeled)) == n_labeled and len(set(unlabeled)) == n_unlabeled
        assert not set(labeled) & set(unlabeled)
        assert len(labeled) + len(unlabeled) == 2975
        assert all(i.startswith("train/") and i.count("/") == 2 for i in labeled + unlabeled)
        union = set(labeled) | set(unlabeled)
        train_ids = train_ids or union
        assert union == train_ids  # every partition splits the same 2975 images
    val = adapter.val_list()
    assert len(val) == 500 and len(set(val)) == 500 and all(i.startswith("val/") for i in val)
    assert not set(val) & train_ids


def test_partition_ids_are_independent_of_the_cache_copy():
    adapter = CityscapesAdapter(Path("/nonexistent"))
    first = adapter.partition("1_30", "labeled")
    first.append("junk")
    assert len(adapter.partition("1_30", "labeled")) == 100
    assert adapter.partition("1_30", "labeled")[0] == "train/weimar/weimar_000085_000019"


def test_split_line_parsing():
    line = "leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png"
    assert split_line_to_id(line) == "train/aachen/aachen_000000_000019"
    assert split_line_to_id("leftImg8bit/val/lindau/lindau_000001_000019_leftImg8bit.png") == "val/lindau/lindau_000001_000019"
    for bad in ("train/aachen/x_leftImg8bit.png", "leftImg8bit/train/aachen/x.png", "leftImg8bit/train/x_leftImg8bit.png"):
        with pytest.raises(ValueError, match="split-list entry"):
            split_line_to_id(bad)


def test_read_id_list_without_final_newline(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("a\n\nb \nc")
    assert read_id_list(path) == ["a", "b", "c"]
    with pytest.raises(FileNotFoundError):
        read_id_list(tmp_path / "missing.txt")


def test_invalid_arguments():
    adapter = CityscapesAdapter(Path("/nonexistent"))
    with pytest.raises(ValueError, match="partition"):
        adapter.partition("1_2", "labeled")
    with pytest.raises(ValueError, match="subset"):
        adapter.partition("1_30", "all")
    with pytest.raises(ValueError, match="split"):
        adapter.image_ids("test")
    for bad in ("aachen_000000_000019", "train/aachen", "trainval/a/b", "train/a/b/c"):
        with pytest.raises(ValueError, match="invalid Cityscapes image id"):
            adapter.image_path(bad)
    with pytest.raises(FileNotFoundError):
        adapter.image_ids("train")


def test_lut_matches_the_official_table():
    lut = labelid_lut()
    assert lut.dtype == np.uint8 and lut.shape == (256,)
    assert {int(l): int(lut[l]) for l in np.flatnonzero(lut != 255)} == C.CITYSCAPES_LABELID_TO_TRAINID
    assert sorted(C.CITYSCAPES_LABELID_TO_TRAINID.values()) == list(range(19))
    ignored = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]  # official ignoreInEval ids
    assert all(lut[i] == 255 for i in ignored) and lut[255] == 255


# ----------------------------------------------------------------------------------------------
# Synthetic Cityscapes folder
# ----------------------------------------------------------------------------------------------
def make_dataset(root: Path, n=2, size=(8, 16)):
    """Tiny folder with the real layout; image k is filled with colour (10 + k, 20, 30) (RGB)."""
    ids = {}
    for split in ("train", "val"):
        ids[split] = []
        for k in range(n):
            stem = f"town_{k:06d}_000019"
            image_id = f"{split}/town/{stem}"
            rgb = np.zeros(size + (3,), np.uint8)
            rgb[:] = (10 + k, 20, 30)
            (root / "leftImg8bit" / split / "town").mkdir(parents=True, exist_ok=True)
            (root / "gtFine" / split / "town").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / "leftImg8bit" / f"{image_id}_leftImg8bit.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            labels = (np.arange(size[0] * size[1]) % 35).reshape(size).astype(np.uint8)
            cv2.imwrite(str(root / "gtFine" / f"{image_id}_gtFine_labelIds.png"), labels)
            ids[split].append(image_id)
    return ids


def test_adapter_on_synthetic_folder(tmp_path):
    ids = make_dataset(tmp_path)
    adapter = CityscapesAdapter(tmp_path)
    assert isinstance(adapter, DatasetAdapter) and adapter.name == "cityscapes" and adapter.root == tmp_path
    assert adapter.class_names == C.CITYSCAPES_CLASSES
    assert adapter.image_ids("train") == ids["train"] and adapter.image_ids("val") == ids["val"]
    image = adapter.load_image(ids["train"][1])
    assert image.shape == (8, 16, 3) and image.dtype == np.uint8 and tuple(image[0, 0]) == (11, 20, 30)  # RGB order
    label = adapter.load_label(ids["val"][0])
    raw = (np.arange(128) % 35).reshape(8, 16)
    expected = np.array([C.CITYSCAPES_LABELID_TO_TRAINID.get(int(v), 255) for v in raw.ravel()], np.uint8).reshape(8, 16)
    assert label.dtype == np.uint8 and np.array_equal(label, expected)
    assert adapter.image_path(ids["train"][0]) == tmp_path / "leftImg8bit/train/town/town_000000_000019_leftImg8bit.png"
    with pytest.raises(FileNotFoundError):
        adapter.load_image("train/town/town_999999_000019")
    with pytest.raises(FileNotFoundError):
        adapter.load_label("train/town/town_999999_000019")


def test_label_with_wrong_format_is_rejected(tmp_path):
    ids = make_dataset(tmp_path, n=1)
    path = tmp_path / "gtFine" / f"{ids['train'][0]}_gtFine_labelIds.png"
    cv2.imwrite(str(path), np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError, match="single-channel uint8"):
        CityscapesAdapter(tmp_path).load_label(ids["train"][0])


def test_verify_reports_problems_on_a_small_folder(tmp_path):
    make_dataset(tmp_path)
    problems = CityscapesAdapter(tmp_path).verify()
    assert any("train: found 2 images, expected 2975" in p for p in problems)
    assert any("val.txt does not list" in p for p in problems)
    assert any(p.startswith("1_30:") for p in problems)


def test_get_adapter_follows_the_config(tmp_path):
    cfg = Config()
    cfg.paths.cityscapes_root = str(tmp_path / "cs")
    adapter = get_adapter(cfg)
    assert isinstance(adapter, CityscapesAdapter) and adapter.root == tmp_path / "cs"
    cfg.data.dataset = "mnist"
    with pytest.raises(ValueError, match="unknown dataset"):
        get_adapter(cfg)


def test_atomic_write(tmp_path):
    target = tmp_path / "a" / "b" / "f.bin"
    atomic_write(target, b"one")
    atomic_write(target, b"two")
    assert target.read_bytes() == b"two" and [p.name for p in target.parent.iterdir()] == ["f.bin"]


def test_prepare_cityscapes_reports_a_bad_folder(tmp_path):
    make_dataset(tmp_path)
    tool = load_tool("prepare_cityscapes")
    assert tool.main(["--root", str(tmp_path)]) == 1
    with pytest.raises(SystemExit, match="outside|inside"):
        tool.main(["--root", str(tmp_path), "--json-out", str(tmp_path / "stats.json")])


# ----------------------------------------------------------------------------------------------
# Real Cityscapes
# ----------------------------------------------------------------------------------------------
@pytest.mark.data
def test_real_dataset_verifies(cityscapes_root):
    adapter = CityscapesAdapter(cityscapes_root)
    assert adapter.verify() == []
    assert len(adapter.image_ids("train")) == 2975 and len(adapter.image_ids("val")) == 500
    assert sorted(adapter.val_list()) == adapter.image_ids("val")


@pytest.mark.data
@pytest.mark.parametrize("partition", sorted(COUNTS))
def test_real_every_listed_image_exists(cityscapes_root, partition):
    adapter = CityscapesAdapter(cityscapes_root)
    ids = adapter.partition(partition, "labeled") + adapter.partition(partition, "unlabeled")
    assert len(ids) == 2975
    assert all(adapter.image_path(i).is_file() and adapter.label_path(i).is_file() for i in ids)


@pytest.mark.data
def test_real_val_images_exist(cityscapes_root):
    adapter = CityscapesAdapter(cityscapes_root)
    assert all(adapter.image_path(i).is_file() and adapter.label_path(i).is_file() for i in adapter.val_list())


@pytest.mark.data
def test_real_image_and_label(cityscapes_root):
    adapter = CityscapesAdapter(cityscapes_root)
    image_id = adapter.partition("1_30", "labeled")[0]
    image, label = adapter.load_image(image_id), adapter.load_label(image_id)
    assert image.shape == (1024, 2048, 3) and image.dtype == np.uint8
    assert label.shape == (1024, 2048) and label.dtype == np.uint8
    values = set(np.unique(label).tolist())
    assert values <= set(range(19)) | {255} and len(values & set(range(19))) >= 5
    raw = cv2.imread(str(adapter.label_path(image_id)), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(label == 255, ~np.isin(raw, list(C.CITYSCAPES_LABELID_TO_TRAINID)))
    bgr = cv2.imread(str(adapter.image_path(image_id)))
    assert np.array_equal(image[..., ::-1], bgr)


@pytest.mark.data
def test_real_prepare_cityscapes_statistics(cityscapes_root, tmp_path, capsys):
    tool = load_tool("prepare_cityscapes")
    out = tmp_path / "stats.json"
    assert tool.main(["--root", str(cityscapes_root), "--partition", "1_30", "--limit", "4", "--json-out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "person" in text and "ignored" in text
    stats = json.loads(out.read_text())
    assert stats["images"] == 4
    total = sum(c["pixels"] for c in stats["classes"].values()) + stats["ignored_pixels"]
    assert total == 4 * 1024 * 2048
    assert stats["classes"]["road"]["pixels"] > 0 and stats["classes"]["road"]["regions"] >= 4
