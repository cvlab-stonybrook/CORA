"""Tests of the pseudo-label module: interchange format, sliding window, UniMatch loading, SemiVL standalone, tool."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from cora import constants as C
from cora.pseudo import io as pio
from cora.pseudo import semivl_standalone as svl
from cora.pseudo import unimatch as um
from cora.pseudo.sliding_window import sliding_window_probs, window_boxes, window_counts, window_starts

REPO = Path(__file__).resolve().parents[1]
TEACHERS = Path(os.environ.get("CORA_ROOT", REPO)) / "pretrained" / "teachers"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("make_pseudo_labels", REPO / "tools" / "make_pseudo_labels.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------------------------
# io
# ----------------------------------------------------------------------------------------------
def test_io_round_trip_nested_ids_and_quantisation(tmp_path):
    rng = np.random.default_rng(0)
    label = rng.integers(0, 19, (12, 20)).astype(np.uint8)
    label[0, :3] = 255
    conf = rng.random((12, 20)).astype(np.float32)
    ids = ["plain", "train/aachen/aachen_000000_000019", "a/b/c/d"]
    for image_id in ids:
        assert not pio.has_prediction(tmp_path, image_id)
        pio.write_prediction(tmp_path, image_id, label, conf)
        assert pio.has_prediction(tmp_path, image_id)
        got_label, got_conf = pio.read_prediction(tmp_path, image_id)
        assert got_label.dtype == np.uint8 and np.array_equal(got_label, label)
        assert got_conf.dtype == np.float32
        stored = np.rint(conf * 255).astype(np.uint8)
        assert np.array_equal(np.rint(got_conf * 255).astype(np.uint8), stored)
        assert np.abs(got_conf - conf).max() <= 0.5 / 255 + 1e-6
    assert (tmp_path / "train/aachen/aachen_000000_000019_label.png").is_file()
    assert (tmp_path / "train/aachen/aachen_000000_000019_conf.png").is_file()
    assert not [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]  # atomic writes leave no temporary files


def test_io_confidence_endpoints_and_threshold():
    conf = np.array([[0.0, 1.0, 0.9, 0.5]], dtype=np.float32)
    assert pio.quantize_confidence(conf).tolist() == [[0, 255, 230, 128]]
    assert pio.quantize_confidence(conf.astype(np.float16)).dtype == np.uint8
    with pytest.raises(ValueError):
        pio.quantize_confidence(np.array([[1.2]], dtype=np.float32))
    with pytest.raises(ValueError):
        pio.quantize_confidence(np.array([[np.nan]], dtype=np.float32))
    with pytest.raises(TypeError):
        pio.quantize_confidence(np.array([[1]], dtype=np.uint8))


def test_io_without_confidence_and_stale_file_removed(tmp_path):
    label = np.zeros((4, 4), np.uint8)
    pio.write_prediction(tmp_path, "x/img", label, np.full((4, 4), 0.7, np.float32))
    assert pio.read_prediction(tmp_path, "x/img")[1] is not None
    pio.write_prediction(tmp_path, "x/img", label, None)
    got_label, got_conf = pio.read_prediction(tmp_path, "x/img")
    assert got_conf is None and np.array_equal(got_label, label)


def test_io_rejects_bad_input(tmp_path):
    label = np.zeros((4, 4), np.uint8)
    for bad_id in ("", "/abs/path", "../escape", "a/../../b", "a\\b", ".", "a/", "a//b", "a/./b", "a/b/.", "./a"):
        with pytest.raises(ValueError):
            pio.write_prediction(tmp_path, bad_id, label, None)
    with pytest.raises(ValueError):
        pio.write_prediction(tmp_path, "i", label.astype(np.int32), None)
    with pytest.raises(ValueError):
        pio.write_prediction(tmp_path, "i", label, np.zeros((3, 3), np.float32))
    with pytest.raises(FileNotFoundError):
        pio.read_prediction(tmp_path, "missing")
    with pytest.raises(FileNotFoundError):
        pio.read_manifest(tmp_path)


def test_manifest_round_trip(tmp_path):
    info = {"method": "unimatch", "partition": "1_16", "counts": {"selected": 3}, "path": tmp_path, "value": np.float32(0.5)}
    pio.write_manifest(tmp_path / "sub", info)
    got = pio.read_manifest(tmp_path / "sub")
    assert got == {"method": "unimatch", "partition": "1_16", "counts": {"selected": 3}, "path": str(tmp_path), "value": 0.5}


# ----------------------------------------------------------------------------------------------
# sliding window
# ----------------------------------------------------------------------------------------------
def test_window_layout_of_the_cityscapes_protocol():
    assert window_starts(1024) == [0, 534]
    assert window_starts(2048) == [0, 534, 1068, 1602]
    boxes = window_boxes(1024, 2048)
    assert len(boxes) == 8
    assert (0, 801, 0, 801) in boxes and (534, 1024, 1602, 2048) in boxes
    sizes = {(r1 - r0, c1 - c0) for r0, r1, c0, c1 in boxes}
    assert sizes == {(801, 801), (801, 446), (490, 801), (490, 446)}  # edge windows are cut, not padded
    counts = window_counts(1024, 2048)
    assert counts.min() == 1 and counts.max() == 4
    assert counts.sum() == sum((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in boxes)
    with pytest.raises(ValueError):
        window_starts(0)


def _dummy_model(num_classes: int, seed: int = 0):
    """Window-dependent network: per-pixel linear map plus a term from the window mean (so overlaps differ)."""
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(num_classes, 3, generator=gen, dtype=torch.float64)
    bias = torch.randn(num_classes, generator=gen, dtype=torch.float64)
    shapes = []

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        shapes.append(tuple(x.shape))
        pixel = torch.einsum("cd,bdhw->bchw", weight, x.double())
        return (pixel + 3.0 * x.double().mean() * bias.view(1, -1, 1, 1)).float()

    return model_fn, weight.numpy(), bias.numpy(), shapes


def _brute_force(image: np.ndarray, weight: np.ndarray, bias: np.ndarray, crop: int, stride: int) -> np.ndarray:
    channels, height, width = weight.shape[0], image.shape[1], image.shape[2]
    total = np.zeros((channels, height, width))
    count = np.zeros((height, width))
    row = 0
    while row < height:
        col = 0
        while col < width:
            win = image[:, row : min(height, row + crop), col : min(width, col + crop)]
            logits = np.einsum("cd,dhw->chw", weight, win) + 3.0 * win.mean() * bias[:, None, None]
            logits -= logits.max(axis=0, keepdims=True)
            exp = np.exp(logits)
            total[:, row : min(height, row + crop), col : min(width, col + crop)] += exp / exp.sum(axis=0, keepdims=True)
            count[row : min(height, row + crop), col : min(width, col + crop)] += 1
            col += stride
        row += stride
    return total / count


@pytest.mark.parametrize("size,crop,stride", [((11, 13), 5, 3), ((20, 35), 9, 6), ((7, 7), 10, 6), ((16, 33), 8, 5)])
def test_sliding_window_matches_brute_force(size, crop, stride):
    model_fn, weight, bias, shapes = _dummy_model(4)
    image = torch.randn(3, *size, generator=torch.Generator().manual_seed(1))
    probs = sliding_window_probs(model_fn, image, 4, crop=crop, stride=stride)
    reference = _brute_force(image.double().numpy(), weight, bias, crop, stride)
    assert probs.shape == (4, *size) and probs.dtype == torch.float32
    np.testing.assert_allclose(probs.numpy(), reference, atol=1e-5)
    np.testing.assert_allclose(probs.sum(0).numpy(), 1.0, atol=1e-5)  # normalised by the window count
    assert len(shapes) == len(window_boxes(*size, crop, stride))
    assert max(s[2] for s in shapes) <= crop and max(s[3] for s in shapes) <= crop


def test_sliding_window_label_equals_unnormalised_argmax_and_validates():
    model_fn, _, _, _ = _dummy_model(3)
    image = torch.randn(3, 14, 17, generator=torch.Generator().manual_seed(2))
    probs = sliding_window_probs(model_fn, image, 3, crop=6, stride=4)
    summed = torch.zeros(3, 14, 17)
    for r0, r1, c0, c1 in window_boxes(14, 17, 6, 4):
        summed[:, r0:r1, c0:c1] += model_fn(image[None, :, r0:r1, c0:c1])[0].softmax(0)
    assert torch.equal(probs.argmax(0), summed.argmax(0))  # the label of the repositories is unchanged
    with pytest.raises(ValueError):
        sliding_window_probs(model_fn, image[None], 3)
    with pytest.raises(ValueError):
        sliding_window_probs(model_fn, image, 5, crop=6, stride=4)  # network returns 3 classes, 5 expected


# ----------------------------------------------------------------------------------------------
# metrics and oracle noise
# ----------------------------------------------------------------------------------------------
def test_stride_larger_than_crop_is_rejected():
    """Regression: with stride > crop some pixels have no window, 0 / 0 gave NaN probabilities."""
    model_fn, _, _, _ = _dummy_model(3)
    with pytest.raises(ValueError, match="stride 8 exceeds crop 5"):
        sliding_window_probs(model_fn, torch.randn(3, 20, 20), 3, crop=5, stride=8)
    with pytest.raises(ValueError):
        window_boxes(20, 20, crop=5, stride=8)
    with pytest.raises(ValueError, match="stride <= crop"):
        svl.sliding_window_probs(model_fn, torch.randn(3, 20, 20), 3, crop=5, stride=8)
    probs = sliding_window_probs(model_fn, torch.randn(3, 20, 20), 3, crop=5, stride=5)  # stride == crop tiles exactly
    assert torch.isfinite(probs).all()


def test_confusion_and_iou(tool):
    target = np.array([[0, 0, 1, 1, 255], [2, 2, 2, 1, 0]], dtype=np.uint8)
    pred = np.array([[0, 1, 1, 1, 0], [2, 2, 0, 1, 0]], dtype=np.uint8)
    confusion = tool.confusion_matrix(pred, target, 3)
    assert confusion.sum() == 9  # the void pixel is ignored
    assert confusion.tolist() == [[2, 1, 0], [0, 3, 0], [1, 0, 2]]
    iou = tool.iou_from_confusion(confusion)
    np.testing.assert_allclose(iou, [100 * 2 / 4, 100 * 3 / 4, 100 * 2 / 3], atol=1e-6)
    assert np.array_equal(svl.confusion_matrix(pred, target, 3), confusion)


def test_oracle_noise_is_deterministic_and_well_formed(tool):
    rng = np.random.default_rng(3)
    label = rng.integers(0, 19, (70, 90)).astype(np.uint8)
    label[:5] = 255
    first = tool.oracle_noise_prediction(label, "train/a/img", 19, seed=0, noise_rate=0.3, cell=16)
    again = tool.oracle_noise_prediction(label, "train/a/img", 19, seed=0, noise_rate=0.3, cell=16)
    other_id = tool.oracle_noise_prediction(label, "train/a/img2", 19, seed=0, noise_rate=0.3, cell=16)
    other_seed = tool.oracle_noise_prediction(label, "train/a/img", 19, seed=1, noise_rate=0.3, cell=16)
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert not np.array_equal(first[0], other_id[0]) and not np.array_equal(first[0], other_seed[0])
    out, conf = first
    assert out.dtype == np.uint8 and conf.dtype == np.float32 and out.shape == conf.shape == label.shape
    assert np.array_equal(out[:5], label[:5]) and (conf[:5] == 0).all()  # void stays void
    assert out[label != 255].max() < 19
    changed = (out != label) & (label != 255)
    assert changed.any() and (conf[changed] < 0.9).all()  # flipped pixels fall below the 0.9 threshold
    assert conf.max() <= 1.0 and conf[label != 255].min() >= 0.3
    clean = tool.oracle_noise_prediction(label, "train/a/img", 19, noise_rate=0.0)
    assert np.array_equal(clean[0], label) and (clean[1][label != 255] >= 0.9).all()
    with pytest.raises(ValueError):
        tool.oracle_noise_prediction(label.astype(np.int64), "x", 19)
    with pytest.raises(ValueError):
        tool.oracle_noise_prediction(label, "x", 19, noise_rate=1.5)


# ----------------------------------------------------------------------------------------------
# UniMatch loading helpers
# ----------------------------------------------------------------------------------------------
def test_strip_module_prefix_and_preprocess():
    state = OrderedDict([("module.backbone.w", 1), ("head.b", 2), ("module.module.x", 3)])
    assert um.strip_module_prefix(state) == {"backbone.w": 1, "head.b": 2, "module.x": 3}
    image = np.zeros((4, 5, 3), np.uint8)
    image[..., 0] = 255
    tensor = um.preprocess_image(image, torch.device("cpu"))
    assert tensor.shape == (3, 4, 5) and tensor.dtype == torch.float32
    np.testing.assert_allclose(tensor[0, 0, 0].item(), (1 - 0.485) / 0.229, rtol=1e-6)
    np.testing.assert_allclose(tensor[1, 0, 0].item(), (0 - 0.456) / 0.224, rtol=1e-6)
    with pytest.raises(ValueError):
        um.preprocess_image(np.zeros((4, 5), np.uint8), torch.device("cpu"))


def test_checkpoint_loading_is_weights_only(tmp_path):
    good = {"model": {"module.w": torch.ones(2)}, "optimizer": {"state": {}}, "epoch": 3, "previous_best": np.float64(76.5)}
    torch.save(good, tmp_path / "good.pth")
    loaded = um.load_checkpoint(tmp_path / "good.pth")
    assert loaded["epoch"] == 3 and float(loaded["previous_best"]) == 76.5
    assert torch.equal(loaded["model"]["module.w"], torch.ones(2))
    torch.save({"model": {}, "extra": np.arange(3)}, tmp_path / "array.pth")  # numpy arrays are not allow-listed
    with pytest.raises(RuntimeError, match="numpy"):
        um.load_checkpoint(tmp_path / "array.pth")
    torch.save({"model": {}, "extra": io.StringIO}, tmp_path / "klass.pth")
    with pytest.raises(RuntimeError, match="StringIO"):
        um.load_checkpoint(tmp_path / "klass.pth")
    torch.save({"weights": 1}, tmp_path / "other.pth")
    with pytest.raises(ValueError, match="no 'model'"):
        um.load_checkpoint(tmp_path / "other.pth")
    with pytest.raises(FileNotFoundError):
        um.load_checkpoint(tmp_path / "absent.pth")


def test_backbone_context_restores_working_directory(tmp_path):
    weights = tmp_path / "backbone.pth"
    torch.save({"a": torch.zeros(1)}, weights)
    before = os.getcwd()
    with um._backbone_in_working_directory(weights):
        assert Path("pretrained/resnet101.pth").resolve() == weights.resolve()
        assert os.getcwd() != before
    assert os.getcwd() == before
    with pytest.raises(RuntimeError):
        with um._backbone_in_working_directory(weights):
            raise RuntimeError("boom")
    assert os.getcwd() == before
    with pytest.raises(FileNotFoundError):
        with um._backbone_in_working_directory(tmp_path / "missing.pth"):
            pass


def test_build_unimatch_needs_a_checkout(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a UniMatch checkout"):
        um.build_unimatch(tmp_path, tmp_path / "backbone.pth")


# ----------------------------------------------------------------------------------------------
# SemiVL standalone (everything that does not need the legacy environment)
# ----------------------------------------------------------------------------------------------
def test_restricted_unpickler_allows_plain_state_dicts_only():
    payload = OrderedDict([("epoch", 3), ("lr", [0.1, 0.2])])
    assert svl.RestrictedUnpickler(io.BytesIO(pickle.dumps(payload))).load() == payload

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo refused",))

    with pytest.raises(pickle.UnpicklingError, match="system"):
        svl.RestrictedUnpickler(io.BytesIO(pickle.dumps(Evil()))).load()
    module = svl._restricted_pickle_module()
    with pytest.raises(pickle.UnpicklingError):
        module.load(io.BytesIO(pickle.dumps(Evil())))


def test_semivl_config_with_python_tuple(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset: cityscapes\nscale_ratio_range: !!python/tuple [0.5, 2.0]\nnclass: 19\nmodel_args: {renorm_clip_img: true}\n")
    cfg = svl.load_config(path)
    assert cfg["scale_ratio_range"] == (0.5, 2.0) and cfg["nclass"] == 19
    path.write_text("x: !!python/object/apply:os.system ['echo no']\n")
    with pytest.raises(Exception):
        svl.load_config(path)
    release = TEACHERS / "semivl" / "semivl_cs_1_16" / "config.yaml"
    if release.is_file():
        released = svl.load_config(release)
        assert released["model"] == "mmseg.vlm-vlg-aspp-s2p4-skr04-ftap-mcvitb" and released["crop_size"] == 801


def test_standalone_writer_matches_cora_io(tmp_path):
    rng = np.random.default_rng(4)
    label = rng.integers(0, 19, (9, 14)).astype(np.uint8)
    conf = rng.random((9, 14)).astype(np.float32)
    svl.write_prediction(tmp_path / "svl", "train/city/img", label, conf)
    pio.write_prediction(tmp_path / "cora", "train/city/img", label, conf)
    a_label, a_conf = pio.read_prediction(tmp_path / "svl", "train/city/img")
    b_label, b_conf = pio.read_prediction(tmp_path / "cora", "train/city/img")
    assert np.array_equal(a_label, b_label) and np.array_equal(a_conf, b_conf)
    assert [p.name for p in sorted((tmp_path / "svl/train/city").iterdir())] == ["img_conf.png", "img_label.png"]
    with pytest.raises(ValueError):
        svl.write_prediction(tmp_path, "../x", label, conf)
    with pytest.raises(ValueError):
        svl.write_prediction(tmp_path, "x", label, conf + 1.0)


def test_standalone_sliding_window_equals_package_implementation():
    model_fn, _, _, _ = _dummy_model(5)
    image = torch.randn(3, 25, 31, generator=torch.Generator().manual_seed(5))
    ours = sliding_window_probs(model_fn, image, 5, crop=10, stride=6)
    theirs = svl.sliding_window_probs(model_fn, image, 5, crop=10, stride=6)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_standalone_preprocess_equals_package_preprocess():
    image = np.random.default_rng(6).integers(0, 256, (6, 7, 3)).astype(np.uint8)
    assert torch.allclose(svl.preprocess(image, torch.device("cpu")), um.preprocess_image(image, torch.device("cpu")))


# ----------------------------------------------------------------------------------------------
# tool
# ----------------------------------------------------------------------------------------------
class _FakeAdapter:
    class_names = ("a", "b", "c")

    def image_ids(self, split):
        return [f"{split}/city/img{i:02d}" for i in range(10)]

    def partition(self, partition, which):
        return [f"train/city/u{i:02d}" for i in range(10)]


def test_select_ids_and_gpu_parsing(tool):
    args = tool.build_parser().parse_args(["--method", "oracle-noise", "--partition", "1_30", "--date", "2026-01-01", "--every", "3", "--limit", "2"])
    assert tool.select_ids(args, _FakeAdapter()) == ["train/city/u00", "train/city/u03"]
    args.split = "val"
    args.limit, args.every = None, 4
    assert tool.select_ids(args, _FakeAdapter()) == ["val/city/img00", "val/city/img04", "val/city/img08"]
    assert tool.parse_gpus("3") == ["3"] and tool.parse_gpus("0, 1,2") == ["0", "1", "2"] and tool.parse_gpus(None) == []
    for bad in ("a", "1,1", "1,,x"):
        with pytest.raises(ValueError):
            tool.parse_gpus(bad)


def _identity(**changes):
    identity = {"method": "unimatch", "partition": "1_16", "checkpoint_sha256": "aaa", "crop": 801, "stride": 534, "noise": None}
    identity.update(changes)
    return identity


def _manifest_of(identity):
    return {"method": identity["method"], "partition": identity["partition"], "checkpoint_sha256": identity["checkpoint_sha256"],
            "window": {"crop": identity["crop"], "stride": identity["stride"]}, "noise": identity["noise"]}


def test_resume_guard_refuses_foreign_folder(tool, tmp_path):
    pio.write_manifest(tmp_path, _manifest_of(_identity()))
    tool._check_resume(tmp_path, _identity(), overwrite=False)
    for changes in ({"method": "semivl"}, {"partition": "1_8"}, {"checkpoint_sha256": "bbb"}, {"crop": 400}, {"stride": 400},
                    {"noise": {"rate": 0.1, "cell": 32, "seed": 0}}):
        with pytest.raises(RuntimeError, match="other settings"):
            tool._check_resume(tmp_path, _identity(**changes), overwrite=False)
    tool._check_resume(tmp_path, _identity(method="semivl"), overwrite=True)


def test_resume_guard_refuses_predictions_without_manifest(tool, tmp_path):
    tool._check_resume(tmp_path / "does_not_exist", _identity(), overwrite=False)
    tool._check_resume(tmp_path, _identity(), overwrite=False)  # empty folder
    pio.write_prediction(tmp_path, "train/city/img", np.zeros((3, 3), np.uint8), None)
    with pytest.raises(RuntimeError, match="no manifest.json"):
        tool._check_resume(tmp_path, _identity(), overwrite=False)
    tool._check_resume(tmp_path, _identity(), overwrite=True)


class _OracleAdapter(_FakeAdapter):
    root = Path("/unused")

    def load_label(self, image_id):
        return np.random.default_rng(len(image_id)).integers(0, 3, (40, 48)).astype(np.uint8)


def test_interrupted_run_leaves_manifest_that_blocks_a_foreign_resume(tool, tmp_path, monkeypatch):
    """Regression: the manifest used to be written at the end only, so a killed run could be resumed by another teacher."""
    import cora.data.adapters as adapters

    monkeypatch.setattr(adapters, "get_adapter", lambda cfg: _OracleAdapter())
    out = tmp_path / "debug" / "1_30"
    argv = ["--method", "oracle-noise", "--partition", "1_30", "--date", "2026-11-19", "--limit", "3", "--out-dir", str(out)]
    original = tool.run_shard

    def killed(args, ids, adapter, out_dir):
        original(args, ids[:2], adapter, out_dir)  # two images are written, then the run dies
        raise KeyboardInterrupt

    monkeypatch.setattr(tool, "run_shard", killed)
    with pytest.raises(KeyboardInterrupt):
        tool.main(argv)
    manifest = pio.read_manifest(out)
    assert manifest["status"] == "in_progress" and manifest["noise"] == {"rate": 0.1, "cell": 32, "seed": 0}
    monkeypatch.setattr(tool, "run_shard", original)
    with pytest.raises(RuntimeError, match="other settings"):
        tool.main(argv + ["--seed", "5"])
    assert pio.read_manifest(out)["noise"]["seed"] == 0  # the refusal left the manifest alone
    assert tool.main(argv) == 0  # the same settings resume: 2 kept, 1 written
    done = pio.read_manifest(out)
    assert done["status"] == "complete" and done["counts"] == {"selected": 3, "present": 3, "written_this_run": 1, "skipped_existing": 2}
    assert done["cora_commit"] is None or len(done["cora_commit"]) == 40
    (out / pio.MANIFEST_NAME).unlink()  # a folder without manifest is not resumed by anything
    with pytest.raises(RuntimeError, match="no manifest.json"):
        tool.main(argv)


def test_eval_report_is_copied_into_the_manifest_and_must_match(tool, tmp_path):
    identity = _identity()
    report = {"method": "unimatch", "partition": "1_16", "checkpoint_sha256": "aaa", "miou": 76.5, "images_evaluated": 500,
              "split": "val", "date": "2026-11-19", "crop": 801, "stride": 534}
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(report))
    assert tool.load_eval_report(path, identity)["miou"] == 76.5
    path.write_text(json.dumps({**report, "checkpoint_sha256": "bbb"}))
    with pytest.raises(ValueError, match="not for this run"):
        tool.load_eval_report(path, identity)
    path.write_text(json.dumps({"method": "unimatch"}))
    with pytest.raises(ValueError, match="not an --evaluate report"):
        tool.load_eval_report(path, identity)


def test_window_settings_are_validated(tool):
    parser = tool.build_parser()
    base = ["--method", "unimatch", "--partition", "1_16", "--date", "2026-11-19"]
    args = parser.parse_args(base)
    tool.resolve_window(args)
    assert (args.crop, args.stride) == (801, 534)
    args = parser.parse_args(base + ["--crop", "400"])
    tool.resolve_window(args)
    assert args.stride == 266  # int(400 * 2 / 3): a lone --crop no longer leaves holes between the windows
    args = parser.parse_args(base + ["--crop", "400", "--stride", "500"])
    with pytest.raises(ValueError, match="stride"):
        tool.resolve_window(args)
    with pytest.raises(SystemExit):
        tool.main(base + ["--crop", "400", "--stride", "500"])


@pytest.mark.data
@pytest.mark.parametrize("workers", [1, 2])
def test_tool_oracle_noise_on_cityscapes(tool, cityscapes_root, tmp_path, workers):
    out = tmp_path / "debug" / "1_30"
    base = ["--method", "oracle-noise", "--partition", "1_30", "--date", "2026-11-19", "--limit", "4", "--workers", str(workers),
            "--set", f"paths.cityscapes_root={cityscapes_root}"]
    argv = base + ["--out-dir", str(out)]
    assert tool.main(argv) == 0
    manifest = pio.read_manifest(out)
    assert manifest["method"] == "debug" and manifest["synthetic"] is True and manifest["date"] == "2026-11-19"
    assert manifest["counts"] == {"selected": 4, "present": 4, "written_this_run": 4, "skipped_existing": 0}
    from cora.config import load_config
    from cora.data.adapters import get_adapter

    adapter = get_adapter(load_config(None, [f"paths.cityscapes_root={cityscapes_root}"]))
    ids = adapter.partition("1_30", "unlabeled")[:4]
    label, conf = pio.read_prediction(out, ids[0])
    truth = adapter.load_label(ids[0])
    assert label.shape == truth.shape == (1024, 2048) and conf is not None
    expected, _ = tool.oracle_noise_prediction(truth, ids[0], 19)
    assert np.array_equal(label, expected)  # same result however the work is sharded
    assert tool.main(argv) == 0  # resume keeps the files
    assert pio.read_manifest(out)["counts"]["skipped_existing"] == 4
    foreign = tmp_path / "unimatch" / "1_30"
    with pytest.raises(ValueError, match="teacher folder"):
        tool.main(base + ["--out-dir", str(foreign)])


# ----------------------------------------------------------------------------------------------
# real checkpoint on GPU
# ----------------------------------------------------------------------------------------------
def _unimatch_teacher_and_adapter(cityscapes_root):
    from cora.config import project_root, load_config
    from cora.data.adapters import get_adapter

    checkpoint = TEACHERS / "unimatch" / "cityscapes_r101_1_16.pth"
    checkout = project_root() / "third_party" / "UniMatch"
    if not checkpoint.is_file() or not checkout.is_dir():
        pytest.skip("UniMatch checkpoint or checkout not available")
    before = os.getcwd()
    teacher = um.UniMatchTeacher(checkout, checkpoint, TEACHERS / "unimatch" / "resnet101.pth", device="cuda")
    assert os.getcwd() == before and abs(teacher.reported_miou - 76.59) < 0.01
    return teacher, get_adapter(load_config(None, [f"paths.cityscapes_root={cityscapes_root}"]))


@pytest.mark.gpu
@pytest.mark.data
def test_unimatch_1_16_miou_on_a_spread_of_validation_images(tool, cityscapes_root):
    """A check of the input pipeline: every 12th validation image (42 images, all cities).

    Measured on GPU: 72.10 mIoU with RGB input and 67.03 with the channels swapped (the full 500 images give 76.54, the
    published value 76.6), so a floor of 70 separates a correct pipeline from a wrong one; pixel accuracy does not (0.966 vs 0.961).
    """
    teacher, adapter = _unimatch_teacher_and_adapter(cityscapes_root)
    ids = adapter.image_ids("val")[::12]
    assert len(ids) >= 40
    confusion = np.zeros((19, 19), dtype=np.int64)
    for image_id in ids:
        label, conf = teacher.predict(adapter.load_image(image_id))
        truth = adapter.load_label(image_id)
        assert label.shape == conf.shape == truth.shape and label.dtype == np.uint8
        assert conf.min() >= 1.0 / 19 - 1e-6 and conf.max() <= 1.0 + 1e-6
        confusion += tool.confusion_matrix(label, truth, 19)
    assert tool.iou_from_confusion(confusion).mean() > 70.0


@pytest.mark.gpu
@pytest.mark.data
def test_unimatch_labels_equal_the_reference_evaluation_loop(cityscapes_root):
    """Labels equal the argmax of UniMatch's own evaluation (``supervised.py``: summed softmax of the unpadded windows)."""
    teacher, adapter = _unimatch_teacher_and_adapter(cityscapes_root)
    for image_id in adapter.image_ids("val")[::250]:
        image = adapter.load_image(image_id)
        label, _ = teacher.predict(image)
        tensor = um.preprocess_image(image, torch.device("cuda"))
        summed = torch.zeros(19, *tensor.shape[1:], device="cuda")
        with torch.no_grad():
            for row in range(0, tensor.shape[1], 534):
                for col in range(0, tensor.shape[2], 534):
                    window = tensor[None, :, row:row + 801, col:col + 801]
                    summed[:, row:row + 801, col:col + 801] += teacher.model(window)[0].softmax(dim=0)
        assert np.array_equal(label, summed.argmax(0).to(torch.uint8).cpu().numpy())


# ----------------------------------------------------------------------------------------------
# SemiVL standalone: driver logic with a stand-in network (the real model needs the legacy environment)
# ----------------------------------------------------------------------------------------------
def test_standalone_source_is_python38_syntax():
    import ast

    source = (REPO / "cora" / "pseudo" / "semivl_standalone.py").read_text()
    ast.parse(source, feature_version=(3, 8))
    assert "from cora" not in source and "import cora" not in source  # runs without the CORA package


def _make_fake_cityscapes(root: Path, ids, size=(40, 56)):
    from PIL import Image

    rng = np.random.default_rng(7)
    label_ids = np.array([7, 8, 26, 0, 33], dtype=np.uint8)  # road, sidewalk, car, unlabeled (void), bicycle
    for image_id in ids:
        image_path = root / "leftImg8bit" / f"{image_id}_leftImg8bit.png"
        label_path = root / "gtFine" / f"{image_id}_gtFine_labelIds.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (*size, 3)).astype(np.uint8)).save(image_path)
        Image.fromarray(label_ids[rng.integers(0, 5, size)]).save(label_path)


def test_standalone_run_with_stand_in_model(tmp_path, monkeypatch):
    ids = [f"train/city/img{i}" for i in range(5)]
    root = tmp_path / "cs"
    _make_fake_cityscapes(root, ids)
    (tmp_path / "ids.txt").write_text("\n".join(ids))  # no final newline, as in the split lists
    torch.manual_seed(0)
    net = torch.nn.Conv2d(3, 19, 1)
    monkeypatch.setattr(svl, "build_semivl", lambda *a: (net.eval(), {}, 7))
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"x")

    def args_for(shard, count, evaluate=False, result="r"):
        argv = ["--semivl-dir", str(tmp_path), "--checkpoint", str(checkpoint), "--cityscapes-root", str(root),
                "--id-file", str(tmp_path / "ids.txt"), "--out-dir", str(tmp_path / "out"), "--device", "cpu",
                "--shard-index", str(shard), "--shard-count", str(count), "--crop", "30", "--stride", "20",
                "--result-json", str(tmp_path / f"{result}{shard}.json")]
        return svl.build_parser().parse_args(argv + (["--evaluate"] if evaluate else []))

    for shard in range(2):
        svl.run(args_for(shard, 2))
    results = [json.loads((tmp_path / f"r{i}.json").read_text()) for i in range(2)]
    assert [r["written"] for r in results] == [3, 2] and results[0]["extra"]["teacher_epoch"] == 7
    assert results[0]["extra"]["checkpoint_sha256"] == svl.sha256_file(checkpoint)
    for image_id in ids:
        image = svl.preprocess(svl.load_rgb(root, image_id), torch.device("cpu"))
        probs = sliding_window_probs(net, image, 19, crop=30, stride=20)
        label, conf = pio.read_prediction(tmp_path / "out", image_id)
        assert np.array_equal(label, probs.argmax(0).numpy().astype(np.uint8))
        assert np.abs(conf - probs.max(0).values.detach().numpy()).max() <= 0.5 / 255 + 1e-6
    svl.run(args_for(0, 2, result="again"))  # resume: everything of the shard exists
    assert json.loads((tmp_path / "again0.json").read_text())["skipped_existing"] == 3
    svl.run(args_for(0, 1, evaluate=True, result="ev"))
    evaluation = json.loads((tmp_path / "ev0.json").read_text())
    confusion = np.array(evaluation["confusion"])
    assert confusion.shape == (19, 19) and confusion.sum() > 0 and evaluation["written"] == 0
    truth = svl.load_train_ids(root, ids[0])
    assert set(np.unique(truth)) <= {0, 1, 13, 18, 255}  # labelIds 7, 8, 26, 33 map to 0, 1, 13, 18; 0 is void


def test_standalone_relative_result_json_is_not_written_into_the_checkout(tmp_path, monkeypatch):
    """Regression: build_semivl changes into the SemiVL checkout, a relative --result-json then landed there."""
    ids = ["train/city/img0"]
    root = tmp_path / "cs"
    _make_fake_cityscapes(root, ids)
    (tmp_path / "ids.txt").write_text(ids[0])
    checkout = tmp_path / "semivl_checkout"
    checkout.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    net = torch.nn.Conv2d(3, 19, 1)

    def fake_build(*_):
        os.chdir(checkout)  # what the real build_semivl does
        return net.eval(), {}, 1

    monkeypatch.chdir(work)
    monkeypatch.setattr(svl, "build_semivl", fake_build)
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"x")
    args = svl.build_parser().parse_args(["--semivl-dir", str(checkout), "--checkpoint", str(checkpoint), "--cityscapes-root", str(root),
                                          "--id-file", str(tmp_path / "ids.txt"), "--out-dir", "out", "--device", "cpu",
                                          "--crop", "30", "--stride", "20", "--result-json", "result.json"])
    svl.run(args)
    assert (work / "result.json").is_file() and not (checkout / "result.json").exists()
    assert (work / "out" / "train" / "city" / "img0_label.png").is_file()
    with pytest.raises(ValueError, match="stride"):
        svl.run(svl.build_parser().parse_args(["--semivl-dir", str(checkout), "--checkpoint", str(checkpoint), "--cityscapes-root", str(root),
                                               "--id-file", str(tmp_path / "ids.txt"), "--out-dir", "out", "--device", "cpu",
                                               "--crop", "20", "--stride", "30", "--result-json", "x.json"]))


def test_standalone_image_ids_follow_the_package_rules(tmp_path):
    for bad in ("", ".", "a/", "a//b", "a/./b", "../x", "/abs"):
        with pytest.raises(ValueError):
            svl.prediction_paths(tmp_path, bad)
    assert svl.prediction_paths(tmp_path, "train/city/x") == pio.prediction_paths(tmp_path, "train/city/x")


def test_tool_command_line_for_semivl_is_accepted_by_the_standalone(tool, tmp_path):
    args = tool.build_parser().parse_args(["--method", "semivl", "--partition", "1_16", "--date", "2026-11-19", "--evaluate",
                                           "--semivl-python", "/x/python", "--semivl-dir", "/x/semivl", "--overwrite"])
    tool.resolve_window(args)
    args.checkpoint = tmp_path / "best.pth"
    args.cityscapes_root = tmp_path
    args.out_dir = tmp_path / "out"
    command = tool._semivl_argv(args, 1, 4, tmp_path, tmp_path / "ids.txt")
    assert command[0] == "/x/python" and Path(command[1]).name == "semivl_standalone.py" and Path(command[1]).is_file()
    parsed = svl.build_parser().parse_args(command[2:])
    assert (parsed.shard_index, parsed.shard_count, parsed.evaluate, parsed.overwrite) == (1, 4, True, True)
    assert (parsed.crop, parsed.stride) == (801, 534)


# ----------------------------------------------------------------------------------------------
# confidence resolution of the 8-bit format
# ----------------------------------------------------------------------------------------------
def test_effective_threshold_is_exact_for_the_default_and_bounded_otherwise():
    assert pio.effective_threshold(0.9) == pytest.approx(0.9, abs=1e-12)
    assert pio.effective_threshold(0.0) == pytest.approx(-0.5 / 255) and pio.effective_threshold(1.0) == pytest.approx(254.5 / 255)
    for threshold in (0.5, 0.8, 0.95, 0.99, 0.123):
        assert abs(pio.effective_threshold(threshold) - threshold) <= 0.5 / 255 + 1e-9
    with pytest.raises(ValueError):
        pio.effective_threshold(1.5)


@pytest.mark.parametrize("threshold", [0.9, 0.95, 0.8, 0.73])
def test_stored_confidence_passes_a_threshold_exactly_where_effective_threshold_says(tmp_path, threshold):
    """The pixels kept by ``read_prediction(...) >= t`` are those with c >= effective_threshold(t); only 0.9 is exact."""
    conf = np.random.default_rng(1).uniform(threshold - 0.02, threshold + 0.02, (64, 64)).astype(np.float32)
    pio.write_prediction(tmp_path, "x", np.zeros((64, 64), np.uint8), conf)
    _, stored = pio.read_prediction(tmp_path, "x")
    cutoff = pio.effective_threshold(threshold)
    margin = np.abs(conf - cutoff) > 1e-5  # ignore pixels within float noise of the boundary
    assert np.array_equal((stored >= threshold)[margin], (conf >= cutoff)[margin])
    if threshold == 0.9:
        assert np.array_equal(stored >= 0.9, conf >= 0.9)
    else:
        assert not np.array_equal(stored >= threshold, conf >= threshold)  # the documented 1/255 resolution


# ----------------------------------------------------------------------------------------------
# UniMatch teacher (stand-in network; the real UniMatch network needs the checkout and a GPU)
# ----------------------------------------------------------------------------------------------
def _stand_in_teacher(monkeypatch, tmp_path, dataset="cityscapes", **kwargs):
    seen = {}
    network = torch.nn.Conv2d(3, 19, 1)

    def fake_build(checkout, backbone, config=None):
        seen["config"] = config
        return network

    monkeypatch.setattr(um, "build_unimatch", fake_build)
    monkeypatch.setattr(um, "load_checkpoint", lambda path: {"model": network.state_dict(), "epoch": 7, "previous_best": 61.5})
    checkpoint = tmp_path / "ckpt.pth"
    checkpoint.write_bytes(b"x")
    return um.UniMatchTeacher(tmp_path, checkpoint, tmp_path, device="cpu", dataset=dataset, **kwargs), seen


def test_cityscapes_teacher_keeps_its_protocol(monkeypatch, tmp_path):
    teacher, seen = _stand_in_teacher(monkeypatch, tmp_path, dataset="cityscapes")
    assert seen["config"]["nclass"] == 19 and seen["config"]["backbone"] == "resnet101"
    assert (teacher.crop, teacher.stride, teacher.num_classes) == (801, 534, 19)
    with pytest.raises(ValueError, match="no UniMatch model configuration"):
        um.model_config("coco")


def test_default_checkpoint_names(tool):
    assert tool.default_checkpoint("unimatch", "1_4").name == "cityscapes_r101_1_4.pth"
    assert tool.default_checkpoint("semivl", "1_4").parts[-2:] == ("semivl_cs_1_4", "best.pth")


def test_tool_writes_and_evaluates_unimatch_pseudo_labels(tool, tmp_path, monkeypatch):
    root = tmp_path / "cs"
    ids = [f"val/city/img{i}" for i in range(3)]
    _make_fake_cityscapes(root, ids)
    teacher, _ = _stand_in_teacher(monkeypatch, tmp_path)
    monkeypatch.setattr(um, "UniMatchTeacher", lambda *a, **k: teacher)
    out = tmp_path / "pseudo" / "unimatch" / "1_4"
    argv = ["--method", "unimatch", "--partition", "1_4", "--date", "2026-11-19", "--device", "cpu", "--split", "val",
            "--set", f"paths.cityscapes_root={root}", "--checkpoint", str(tmp_path / "ckpt.pth"), "--out-dir", str(out)]
    assert tool.main(argv) == 0
    manifest = pio.read_manifest(out)
    assert manifest["dataset"] == "cityscapes" and manifest["window"]["crop"] == 801 and manifest["label_ids"] == "Cityscapes trainIds 0..18, 255 = void"
    assert manifest["counts"]["present"] == len(ids)
    label, confidence = pio.read_prediction(out, ids[0])
    assert label.shape == (40, 56) and label.max() < 19 and confidence.shape == label.shape
    report = tmp_path / "eval.json"
    assert tool.main(argv[:-2] + ["--evaluate", "--report", str(report)]) == 0
    result = json.loads(report.read_text())
    assert sorted(result["iou_per_class"]) == sorted(C.CITYSCAPES_CLASSES) and result["images_evaluated"] == len(ids)
