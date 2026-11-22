#!/usr/bin/env python
"""Write pseudo-labels (label PNG + uint8 confidence PNG per image) for the unlabeled images of a partition.

Paper Sec. 4.1: the pseudo-labels of CORA come from UniMatch and SemiVL trained on the same partition.

    python tools/make_pseudo_labels.py --method unimatch --partition 1_16 --date 2026-11-19 [--gpus 3] [--limit N]
    python tools/make_pseudo_labels.py --method unimatch --partition 1_16 --date 2026-11-19 --evaluate
    python tools/make_pseudo_labels.py --method semivl   --partition 1_30 --date 2026-11-19 --gpus 0,1,2,3
    python tools/make_pseudo_labels.py --method oracle-noise --partition 1_30 --date 2026-11-19 --workers 4

Methods
  unimatch      UniMatch DeepLabV3+ ResNet-101 (third_party/UniMatch checkout), runs in this environment.
                Cityscapes (19 classes, 801 px windows).
  semivl        SemiVL (needs its own legacy environment, scripts/pseudo/setup_semivl_env.sh); this tool starts
                cora/pseudo/semivl_standalone.py with that environment's python, one process per GPU.
                Cityscapes only: SemiVL's class names and CLIP text embeddings are hard-wired to Cityscapes.
  oracle-noise  ground truth with synthetic label noise and a matching confidence map, written under the method
                name "debug" for pipeline tests only; it is not a teacher.

Output goes to ``Config.pseudo_dir`` (data/<dataset>/pseudo/<method>/<partition>) unless ``--out-dir`` is given:
``<image_id>_label.png``, ``<image_id>_conf.png`` and ``manifest.json``. Existing predictions are kept
(resume) unless ``--overwrite`` is given. ``--evaluate`` writes nothing: it computes the mIoU of the teacher on
the validation images against the ground truth, so that a teacher can be checked before it is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cora import constants as C  # noqa: E402
from cora.config import load_config, project_root  # noqa: E402
from cora.pseudo import io as pseudo_io  # noqa: E402
from cora.pseudo.unimatch import DATASET_WINDOWS  # noqa: E402

logger = logging.getLogger("make_pseudo_labels")

METHODS = ("unimatch", "semivl", "oracle-noise")
METHOD_DIRS = {"unimatch": "unimatch", "semivl": "semivl", "oracle-noise": "debug"}  # Config.data.ssl_method
IGNORE = C.IGNORE_LABEL
ORACLE_NOTE = "synthetic: ground truth with cell-wise random class flips; never a teacher"
SEMIVL_STANDALONE = Path(__file__).resolve().parents[1] / "cora" / "pseudo" / "semivl_standalone.py"


# ----------------------------------------------------------------------------------------------
# Metrics (mIoU as in UniMatch's ``intersectionAndUnion``: per-class IoU over the whole set, mean over classes)
# ----------------------------------------------------------------------------------------------
def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """(C, C) int64 counts, rows = ground truth, columns = prediction; void pixels (255) are ignored."""
    if pred.shape != target.shape:
        raise ValueError(f"prediction {pred.shape} and target {target.shape} differ in shape")
    valid = (target < num_classes) & (pred < num_classes)
    index = target[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    return np.bincount(index, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def iou_from_confusion(confusion: np.ndarray) -> np.ndarray:
    """Per-class IoU in percent: TP / (TP + FP + FN + 1e-10)."""
    tp = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - tp
    return tp / (union + 1e-10) * 100.0


# ----------------------------------------------------------------------------------------------
# Oracle with synthetic noise (pipeline tests only)
# ----------------------------------------------------------------------------------------------
def oracle_noise_prediction(
    label: np.ndarray, image_id: str, num_classes: int, seed: int = 0, noise_rate: float = 0.1, cell: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Ground-truth label map with random class flips on square cells and a matching confidence map.

    A fraction ``noise_rate`` of the ``cell`` x ``cell`` cells is reassigned to a random class (void pixels
    stay void). Confidence is drawn per cell: 0.90 + 0.10 u in clean cells, 0.30 + 0.55 u in flipped cells (u uniform in
    [0, 1)), 0 on void pixels, so that a 0.9 threshold removes the flipped cells. Deterministic in
    (seed, image_id), independent of processing order and sharding.
    """
    if label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError(f"label must be a 2-D uint8 array, got {label.dtype} with shape {label.shape}")
    if not 0.0 <= noise_rate <= 1.0 or cell < 1:
        raise ValueError(f"invalid noise settings: noise_rate={noise_rate}, cell={cell}")
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    height, width = label.shape
    grid = (-(-height // cell), -(-width // cell))
    flipped = rng.random(grid) < noise_rate
    new_class = rng.integers(0, num_classes, grid).astype(np.uint8)
    uniform = rng.random(grid)

    def to_pixels(cells: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(cells, cell, axis=0), cell, axis=1)[:height, :width]

    valid = label != IGNORE
    flipped_px = to_pixels(flipped)
    out = label.copy()
    changed = flipped_px & valid
    out[changed] = to_pixels(new_class)[changed]
    u_px = to_pixels(uniform)
    confidence = np.where(flipped_px, 0.30 + 0.55 * u_px, 0.90 + 0.10 * u_px).astype(np.float32)
    confidence[~valid] = 0.0
    return out, np.minimum(confidence, 1.0)


# ----------------------------------------------------------------------------------------------
# Predictors: image id -> (label uint8, confidence float32)
# ----------------------------------------------------------------------------------------------
Predictor = Callable[[str], Tuple[np.ndarray, Optional[np.ndarray]]]


def build_predictor(args: argparse.Namespace, adapter) -> Tuple[Predictor, dict]:
    """Predictor of ``args.method`` (unimatch | oracle-noise) and the manifest fields describing it."""
    if args.method == "oracle-noise":
        num_classes = len(adapter.class_names)

        def oracle(image_id: str):
            return oracle_noise_prediction(adapter.load_label(image_id), image_id, num_classes, args.seed, args.noise_rate, args.noise_cell)

        return oracle, {
            "synthetic": True,
            "note": ORACLE_NOTE,
            "noise": {"rate": args.noise_rate, "cell": args.noise_cell, "seed": args.seed},
        }
    if args.method == "unimatch":
        from cora.pseudo.unimatch import UniMatchTeacher

        teacher = UniMatchTeacher(
            Path(args.unimatch_dir), Path(args.checkpoint), Path(args.backbone), device=args.device, crop=args.crop,
            stride=args.stride, dataset=args.dataset
        )
        return (lambda image_id: teacher.predict(adapter.load_image(image_id))), {
            "teacher_epoch": teacher.epoch,
            "teacher_reported_val_miou": round(teacher.reported_miou, 3),
        }
    raise ValueError(f"no in-process predictor for method {args.method!r}")


# ----------------------------------------------------------------------------------------------
# One shard of work (in this process)
# ----------------------------------------------------------------------------------------------
def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_shard(args: argparse.Namespace, ids: Sequence[str], adapter, out_dir: Path) -> dict:
    """Predict ``ids`` and write them (or accumulate the confusion matrix with ``--evaluate``)."""
    predictor, extra = build_predictor(args, adapter)
    num_classes = len(adapter.class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    written = skipped = 0
    seconds: List[float] = []
    for position, image_id in enumerate(ids, 1):
        if not args.evaluate and not args.overwrite and _complete(out_dir, image_id):
            skipped += 1
            continue
        start = time.perf_counter()
        label, confidence = predictor(image_id)
        if args.evaluate:
            confusion += confusion_matrix(label, adapter.load_label(image_id), num_classes)
        else:
            pseudo_io.write_prediction(out_dir, image_id, label, confidence)
            written += 1
        if args.method != "oracle-noise":
            _sync()
        seconds.append(time.perf_counter() - start)
        if position % 25 == 0 or position == len(ids):
            logger.info("shard %d/%d: %d/%d images, %.2f s per image", args.shard_index + 1, args.shard_count, position, len(ids), np.mean(seconds))
    return {
        "images": len(ids),
        "written": written,
        "skipped_existing": skipped,
        "seconds_per_image": seconds,
        "confusion": confusion.tolist() if args.evaluate else None,
        "extra": extra,
    }


def _complete(out_dir: Path, image_id: str) -> bool:
    """Label and confidence file both exist (the label file is written last, see :mod:`cora.pseudo.io`)."""
    label_path, conf_path = pseudo_io.prediction_paths(out_dir, image_id)
    return label_path.is_file() and conf_path.is_file()


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------
def parse_gpus(text: Optional[str]) -> List[str]:
    if not text:
        return []
    gpus = [item.strip() for item in text.split(",") if item.strip()]
    if not gpus or not all(item.isdigit() for item in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError(f"--gpus expects distinct GPU indices such as '3' or '0,1,2,3', got {text!r}")
    return gpus


def select_ids(args: argparse.Namespace, adapter) -> List[str]:
    """Image ids of the requested split, thinned by ``--every`` and cut to ``--limit``."""
    if args.split == "val":
        ids = adapter.image_ids("val")
    else:
        ids = adapter.partition(args.partition, "unlabeled")
    ids = ids[:: args.every]
    return ids[: args.limit] if args.limit is not None else ids


def default_checkpoint(method: str, partition: str, dataset: str = "cityscapes") -> Path:
    teachers = project_root() / "pretrained" / "teachers"
    if method == "unimatch":
        return teachers / "unimatch" / f"{dataset}_r101_{partition}.pth"
    return teachers / "semivl" / f"semivl_cs_{partition}" / "best.pth"


def _worker_argv(args: argparse.Namespace, index: int, count: int, work_dir: Path) -> List[str]:
    """Command-line of a unimatch / oracle-noise worker process (same tool, one shard)."""
    argv = ["--method", args.method, "--partition", args.partition, "--dataset", args.dataset, "--split", args.split,
            "--date", args.date, "--device", args.device, "--crop", str(args.crop), "--stride", str(args.stride),
            "--every", str(args.every), "--seed", str(args.seed), "--noise-rate", str(args.noise_rate),
            "--noise-cell", str(args.noise_cell), "--unimatch-dir", str(args.unimatch_dir), "--backbone", str(args.backbone),
            "--out-dir", str(args.out_dir), "--shard-index", str(index), "--shard-count", str(count), "--work-dir", str(work_dir)]
    if args.checkpoint:
        argv += ["--checkpoint", str(args.checkpoint)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.config:
        argv += ["--config", args.config]
    for item in args.overrides:
        argv += ["--set", item]
    for flag, enabled in (("--evaluate", args.evaluate), ("--overwrite", args.overwrite)):
        if enabled:
            argv.append(flag)
    return argv


def _semivl_argv(args: argparse.Namespace, index: int, count: int, work_dir: Path, id_file: Path) -> List[str]:
    """Command-line of a SemiVL worker (the standalone script in SemiVL's own environment)."""
    argv = [str(args.semivl_python), str(SEMIVL_STANDALONE), "--semivl-dir", str(args.semivl_dir),
            "--checkpoint", str(args.checkpoint), "--cityscapes-root", str(args.cityscapes_root),
            "--id-file", str(id_file), "--out-dir", str(args.out_dir), "--shard-index", str(index),
            "--shard-count", str(count), "--result-json", str(work_dir / f"shard_{index}.json"),
            "--crop", str(args.crop), "--stride", str(args.stride)]
    if args.semivl_config:
        argv += ["--config", str(args.semivl_config)]
    for flag, enabled in (("--evaluate", args.evaluate), ("--overwrite", args.overwrite)):
        if enabled:
            argv.append(flag)
    return argv


def launch_workers(commands: Sequence[List[str]], gpus: Sequence[str]) -> None:
    """Run worker commands in parallel (worker i sees GPU ``gpus[i]`` only) and fail if any of them fails."""
    processes = []
    for index, command in enumerate(commands):
        env = dict(os.environ)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[index]
        logger.info("starting worker %d/%d%s", index + 1, len(commands), f" on GPU {gpus[index]}" if gpus else "")
        processes.append(subprocess.Popen(command, env=env))
    failed = [i for i, process in enumerate(processes) if process.wait() != 0]
    if failed:
        raise RuntimeError(f"worker(s) {failed} failed; see their output above (finished images are kept, rerun to resume)")


def _git_commit(folder: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_identity(args: argparse.Namespace, checkpoint_sha: Optional[str]) -> dict:
    """What makes two runs write interchangeable predictions: teacher, partition, window and noise settings."""
    teacher = args.method != "oracle-noise"
    return {
        "method": METHOD_DIRS[args.method],
        "partition": args.partition,
        "checkpoint_sha256": checkpoint_sha,
        "crop": args.crop if teacher else None,
        "stride": args.stride if teacher else None,
        "noise": None if teacher else {"rate": args.noise_rate, "cell": args.noise_cell, "seed": args.seed},
    }


def _identity_of_manifest(manifest: dict) -> dict:
    window = manifest.get("window") or {}
    return {
        "method": manifest.get("method"),
        "partition": manifest.get("partition"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "crop": window.get("crop"),
        "stride": window.get("stride"),
        "noise": manifest.get("noise"),
    }


def _check_resume(out_dir: Path, identity: dict, overwrite: bool) -> None:
    """Refuse to continue a folder written by another teacher, partition, window or noise setting.

    A folder that holds predictions but no manifest (written by an older version, or by hand) cannot be
    attributed to any run, so it is refused as well. ``--overwrite`` bypasses the guard; the caller then
    replaces the manifest, and images outside the current selection keep their old content.
    """
    manifest_path = out_dir / pseudo_io.MANIFEST_NAME
    if overwrite:
        if manifest_path.is_file() and _identity_of_manifest(pseudo_io.read_manifest(out_dir)) != identity:
            logger.warning("--overwrite: %s was written with other settings; images outside this selection keep their old content", out_dir)
        return
    if not manifest_path.is_file():
        if out_dir.is_dir() and next(out_dir.rglob(f"*{pseudo_io.LABEL_SUFFIX}"), None) is not None:
            raise RuntimeError(f"{out_dir} holds predictions but no {pseudo_io.MANIFEST_NAME}, so their origin is unknown; "
                               "use another --out-dir or pass --overwrite")
        return
    old = _identity_of_manifest(pseudo_io.read_manifest(out_dir))
    differing = sorted(key for key in identity if old[key] != identity[key])
    if differing:
        details = "; ".join(f"{key}: folder {old[key]!r}, this run {identity[key]!r}" for key in differing)
        raise RuntimeError(f"{out_dir} was written with other settings ({details}); use another --out-dir or pass --overwrite")


def load_eval_report(path: Path, identity: dict) -> dict:
    """Validation result of ``--evaluate --report`` for the manifest; refuses a report of another teacher."""
    report = json.loads(Path(path).read_text())
    for key in ("method", "partition", "checkpoint_sha256", "miou"):
        if key not in report:
            raise ValueError(f"{path} is not an --evaluate report (no {key!r})")
    if (METHOD_DIRS.get(report["method"], report["method"]), report["partition"], report["checkpoint_sha256"]) != (
            identity["method"], identity["partition"], identity["checkpoint_sha256"]):
        raise ValueError(f"{path} was produced for {report['method']}/{report['partition']} (checkpoint sha256 "
                         f"{report['checkpoint_sha256']}), not for this run")
    return {key: report.get(key) for key in ("miou", "images_evaluated", "split", "date", "crop", "stride")}


def report_evaluation(confusion: np.ndarray, class_names: Sequence[str]) -> dict:
    iou = iou_from_confusion(confusion)
    for name, value in zip(class_names, iou):
        logger.info("  IoU %-14s %6.2f", name, value)
    logger.info("mIoU %.2f over %d classes", float(iou.mean()), len(iou))
    return {"miou": float(iou.mean()), "iou_per_class": {n: float(v) for n, v in zip(class_names, iou)}}


def orchestrate(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config, [f"data.dataset={args.dataset}", f"data.partition={args.partition}",
                                    f"data.ssl_method={METHOD_DIRS[args.method]}", *args.overrides])
    from cora.data.adapters import get_adapter

    adapter = get_adapter(cfg)
    args.cityscapes_root = adapter.root
    if args.method != "oracle-noise":
        args.checkpoint = args.checkpoint or default_checkpoint(args.method, args.partition, args.dataset)
        if not Path(args.checkpoint).is_file():
            if args.method == "unimatch" and args.partition == "1_30":
                hint = "UniMatch publishes no 1/30 checkpoint; train one with UniMatch's own training code and pass --checkpoint"
            else:
                hint = "run scripts/pseudo/download_teacher_checkpoints.sh"
            raise FileNotFoundError(f"checkpoint not found: {args.checkpoint} ({hint})")
        args.checkpoint = Path(args.checkpoint)
    args.out_dir = Path(args.out_dir) if args.out_dir else cfg.pseudo_dir
    if args.method == "oracle-noise" and {"unimatch", "semivl"} & set(args.out_dir.parts):
        raise ValueError(f"{args.out_dir} looks like a teacher folder; synthetic labels go under pseudo/debug/")

    ids = select_ids(args, adapter)
    if not ids:
        raise ValueError("no images selected")
    gpus = parse_gpus(args.gpus)
    workers = len(gpus) if gpus else args.workers
    workers = max(1, min(workers, len(ids)))
    sha = None
    if args.method != "oracle-noise":
        from cora.pseudo.unimatch import sha256_file

        sha = sha256_file(args.checkpoint)
    identity = run_identity(args, sha)
    validation = load_eval_report(args.eval_report, identity) if args.eval_report else None
    previous_seconds = None
    if not args.evaluate:
        _check_resume(args.out_dir, identity, args.overwrite)
        old_manifest = args.out_dir / pseudo_io.MANIFEST_NAME
        if old_manifest.is_file():
            previous_seconds = pseudo_io.read_manifest(args.out_dir).get("seconds_per_image")
        # Written before the first image, so that an interrupted run still leaves a folder the guard can check.
        pseudo_io.write_manifest(args.out_dir, build_manifest(args, sha, validation, status="in_progress"))
    logger.info("%s on %d %s images of %s (%s), %d worker(s) -> %s", args.method, len(ids), args.split, args.dataset,
                args.partition, workers, "evaluation only" if args.evaluate else args.out_dir)

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pseudo_work_") as work:
        work_dir = Path(work)
        if args.method == "semivl":
            id_file = work_dir / "ids.txt"
            id_file.write_text("\n".join(ids) + "\n")
            launch_workers([_semivl_argv(args, i, workers, work_dir, id_file) for i in range(workers)], gpus)
            shards = [json.loads((work_dir / f"shard_{i}.json").read_text()) for i in range(workers)]
        elif workers == 1 and not gpus:
            shards = [run_shard(_as_shard(args, 0, 1), ids, adapter, args.out_dir)]
        else:
            launch_workers([[sys.executable, str(Path(__file__).resolve()), *_worker_argv(args, i, workers, work_dir)] for i in range(workers)], gpus)
            shards = [json.loads((work_dir / f"shard_{i}.json").read_text()) for i in range(workers)]
    wall = time.perf_counter() - start

    seconds = [s for shard in shards for s in shard["seconds_per_image"]]
    result = {
        "images_selected": len(ids),
        "written": sum(s["written"] for s in shards),
        "skipped_existing": sum(s["skipped_existing"] for s in shards),
        "wall_seconds": round(wall, 2),
        "mean_seconds_per_image_per_worker": round(float(np.mean(seconds)), 3) if seconds else None,
    }
    extra = next((s["extra"] for s in shards if s.get("extra")), {})
    if args.evaluate:
        confusion = np.sum([np.array(s["confusion"], dtype=np.int64) for s in shards], axis=0)
        result.update(report_evaluation(confusion, adapter.class_names))
        result["images_evaluated"] = int(sum(s["images"] - s["skipped_existing"] for s in shards))
        result.update(method=args.method, partition=args.partition, split=args.split, date=args.date, checkpoint_sha256=sha,
                      crop=args.crop, stride=args.stride, every=args.every, limit=args.limit)
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    present = sum(_complete(args.out_dir, i) for i in ids)
    manifest = build_manifest(args, sha, validation, status="complete")
    manifest.update(
        counts={"selected": len(ids), "present": present, "written_this_run": result["written"], "skipped_existing": result["skipped_existing"]},
        limit=args.limit,
        every=args.every,
        seconds_per_image=result["mean_seconds_per_image_per_worker"] if result["mean_seconds_per_image_per_worker"] is not None else previous_seconds,
        **extra,
    )
    pseudo_io.write_manifest(args.out_dir, manifest)
    logger.info("wrote %d, skipped %d existing; %d/%d present; manifest %s", result["written"], result["skipped_existing"], present, len(ids),
                args.out_dir / pseudo_io.MANIFEST_NAME)
    return result


def build_manifest(args: argparse.Namespace, sha: Optional[str], validation: Optional[dict], status: str) -> dict:
    """Fields of ``manifest.json`` that are known before the first image is processed."""
    return {
        "status": status,
        "method": METHOD_DIRS[args.method],
        "tool_method": args.method,
        "dataset": args.dataset,
        "partition": args.partition,
        "split": args.split,
        "date": args.date,
        "checkpoint": str(args.checkpoint) if args.method != "oracle-noise" else None,
        "checkpoint_sha256": sha,
        "window": {"crop": args.crop, "stride": args.stride, "padding": "none (edge windows are cut)", "normalisation": "window-count-normalised softmax"} if args.method != "oracle-noise" else None,
        "confidence": "maximum of the window-count-normalised softmax, stored as uint8 round(c * 255)",
        "label_ids": "Cityscapes trainIds 0..18, 255 = void",
        "noise": {"rate": args.noise_rate, "cell": args.noise_cell, "seed": args.seed} if args.method == "oracle-noise" else None,
        "validation": validation,
        "unimatch_commit": _git_commit(Path(args.unimatch_dir)) if args.method == "unimatch" else None,
        "semivl_commit": _git_commit(Path(args.semivl_dir)) if args.method == "semivl" else None,
        "cora_commit": _git_commit(project_root()),
    }


def _as_shard(args: argparse.Namespace, index: int, count: int) -> argparse.Namespace:
    args.shard_index, args.shard_count = index, count
    return args


def worker_main(args: argparse.Namespace) -> None:
    """Shard worker: process ``ids[shard_index::shard_count]`` and write ``shard_<i>.json`` into ``--work-dir``."""
    cfg = load_config(args.config, [f"data.dataset={args.dataset}", f"data.partition={args.partition}",
                                    f"data.ssl_method={METHOD_DIRS[args.method]}", *args.overrides])
    from cora.data.adapters import get_adapter

    adapter = get_adapter(cfg)
    args.checkpoint = Path(args.checkpoint) if args.checkpoint else (
        default_checkpoint(args.method, args.partition, args.dataset) if args.method != "oracle-noise" else None)
    ids = select_ids(args, adapter)[args.shard_index :: args.shard_count]
    result = run_shard(args, ids, adapter, Path(args.out_dir))
    (Path(args.work_dir) / f"shard_{args.shard_index}.json").write_text(json.dumps(result))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--partition", required=True, help="labeled fraction of the training set, e.g. 1_16")
    p.add_argument("--date", required=True, help="date recorded in the manifest, YYYY-MM-DD (passed by the caller)")
    p.add_argument("--dataset", default="cityscapes", choices=C.DATASETS)
    p.add_argument("--split", default="unlabeled", choices=("unlabeled", "val"), help="images to process (default: the unlabeled images of the partition)")
    p.add_argument("--limit", type=int, default=None, help="process only the first N selected images")
    p.add_argument("--every", type=int, default=1, help="keep every K-th image before --limit (spreads a subset over all cities)")
    p.add_argument("--evaluate", action="store_true", help="compute mIoU on the validation images against the ground truth; writes no pseudo-labels")
    p.add_argument("--report", default=None, help="JSON file for the --evaluate result")
    p.add_argument("--eval-report", default=None, help="--report file of an --evaluate run of the same teacher; its mIoU is copied into manifest.json")
    p.add_argument("--gpus", default=None, help="physical GPU indices, one worker process per GPU, e.g. 3 or 0,1,2,3")
    p.add_argument("--workers", type=int, default=1, help="worker processes without --gpus (CPU shards; oracle-noise)")
    p.add_argument("--device", default="cuda", help="torch device of the in-process unimatch teacher")
    p.add_argument("--overwrite", action="store_true", help="recompute images that already have a prediction (default: resume)")
    p.add_argument("--out-dir", default=None, help="default: Config.pseudo_dir")
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config override")
    p.add_argument("--checkpoint", default=None, help="teacher checkpoint (default: $CORA_ROOT/pretrained/teachers/...)")
    p.add_argument("--unimatch-dir", default=str(project_root() / "third_party" / "UniMatch"))
    p.add_argument("--backbone", default=str(project_root() / "pretrained" / "teachers" / "unimatch" / "resnet101.pth"))
    p.add_argument("--semivl-dir", default=str(project_root() / "third_party" / "semivl"))
    p.add_argument("--semivl-python", default=str(project_root() / "envs" / "semivl" / "bin" / "python"))
    p.add_argument("--semivl-config", default=None, help="default: config.yaml next to the checkpoint")
    p.add_argument("--crop", type=int, default=None, help="window size (default: 801)")
    p.add_argument("--stride", type=int, default=None,
                   help="window stride (default: int(crop * 2 / 3) = 534 for the Cityscapes crop 801)")
    p.add_argument("--seed", type=int, default=0, help="oracle-noise seed")
    p.add_argument("--noise-rate", type=float, default=0.1, help="oracle-noise: fraction of flipped cells")
    p.add_argument("--noise-cell", type=int, default=32, help="oracle-noise: cell size in pixels")
    p.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--work-dir", default=None, help=argparse.SUPPRESS)
    return p


def resolve_window(args: argparse.Namespace) -> None:
    """Fill in the dataset's window defaults and check ``1 <= stride <= crop`` (else pixels stay uncovered).

    Cityscapes: crop 801, stride ``int(crop * 2 / 3)`` (534).
    """
    default_crop, default_stride = DATASET_WINDOWS[args.dataset]
    if args.crop is None:
        args.crop = default_crop
        if args.stride is None:
            args.stride = default_stride
    if args.crop < 1:
        raise ValueError(f"--crop must be positive, got {args.crop}")
    if args.stride is None:
        args.stride = max(1, args.crop * 2 // 3)  # 534 for crop 801
    if not 1 <= args.stride <= args.crop:
        raise ValueError(f"--stride must lie in [1, --crop], got stride={args.stride}, crop={args.crop}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.evaluate:
        args.split = "val"
    if args.every < 1 or (args.limit is not None and args.limit < 1) or args.workers < 1:
        raise SystemExit("--every, --limit and --workers must be positive")
    try:
        resolve_window(args)
    except ValueError as error:
        raise SystemExit(str(error))
    if args.shard_index is not None:
        worker_main(args)
        return 0
    args.shard_index, args.shard_count = 0, 1
    orchestrate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
