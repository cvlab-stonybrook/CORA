"""Checkpoints: only trainable tensors and plain Python values, so they load with ``weights_only=True``.

Layout of a run folder::

    <run_dir>/config.yaml               configuration of the first invocation (later ones differing: config.<time>.yaml)
    <run_dir>/<stage>/last.pt           resumable state (model, optimizer, scheduler, token bank, teacher, RNG, step)
    <run_dir>/<stage>/final.pt          model weights at the end of the stage
    <run_dir>/<stage>/final.pt.pruned   marker left when final.pt was deleted because a later stage supersedes it
    <run_dir>/<stage>/fingerprint.json  the settings that decide this stage's weights (see :mod:`cora.engine.preflight`)
    <run_dir>/<stage>/metrics.json      validation results at the end of the stage
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

logger = logging.getLogger("cora")

FINAL = "final.pt"
LAST = "last.pt"
PRUNED = "final.pt.pruned"
FINGERPRINT = "fingerprint.json"
METRICS = "metrics.json"


class CheckpointMismatch(ValueError):
    """A checkpoint does not fit the model that is built from the current configuration."""


def atomic_save(obj: Dict[str, Any], path: Path) -> None:
    """Write ``obj`` next to ``path`` and rename, so an interrupted save never leaves a broken file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        # e.g. a full disk: the partial file would keep the space that the next attempt needs
        tmp.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def save_final(path: Path, model, meta: Dict[str, Any]) -> None:
    atomic_save({"model": model.trainable_state_dict(), "meta": meta}, path)


def save_last(path: Path, model, optimizer, scheduler, bank, teacher, meta: Dict[str, Any],
              rng: Optional[List[dict]] = None) -> None:
    """Resumable state. ``rng`` holds one :func:`cora.engine.dist.capture_rng_state` result per rank."""
    state: Dict[str, Any] = {
        "model": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "meta": meta,
    }
    if bank is not None:
        state["bank"] = {k: v.detach().cpu() for k, v in bank.state_dict().items()}
    if teacher is not None:
        state["teacher"] = teacher.state_dict()
    if rng is not None:
        state["rng"] = rng
    atomic_save(state, path)


def find_final(run_dir: Path, stage_name: str) -> Optional[Path]:
    path = Path(run_dir) / stage_name / FINAL
    return path if path.is_file() else None


def find_last(run_dir: Path, stage_name: str) -> Optional[Path]:
    path = Path(run_dir) / stage_name / LAST
    return path if path.is_file() else None


def is_pruned(run_dir: Path, stage_name: str) -> bool:
    return (Path(run_dir) / stage_name / PRUNED).is_file()


def is_finished(run_dir: Path, stage_name: str) -> bool:
    """A stage is finished when its ``final.pt`` exists or was pruned after a later stage superseded it."""
    return find_final(run_dir, stage_name) is not None or is_pruned(run_dir, stage_name)


def clean_stale_tmp(stage_dir: Path) -> List[Path]:
    """Remove ``*.tmp`` files that an interrupted :func:`atomic_save` left behind."""
    removed = []
    for path in Path(stage_dir).glob("*.tmp"):
        path.unlink()
        removed.append(path)
    return removed


def read_json(path: Path) -> Optional[dict]:
    path = Path(path)
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------
# Loading into a model
# ----------------------------------------------------------------------------------------------
def load_state_checked(model, state: Dict[str, torch.Tensor], source: str) -> int:
    """Copy a trainable state into ``model`` and return how many tensors were loaded.

    Raises :class:`CheckpointMismatch` when a tensor has another shape than the model's (the checkpoint was
    trained with another model configuration, e.g. ``model.lora_r``) or when no tensor belongs to the model.
    """
    params = dict(model.named_parameters())
    loaded = [name for name in state if name in params]
    if not loaded:
        raise CheckpointMismatch(
            f"{source}: none of its {len(state)} tensors is a parameter of this model; the checkpoint was trained "
            f"with another model configuration. Use the config.yaml of the run that wrote it."
        )
    for name in loaded:
        if params[name].shape != state[name].shape:
            raise CheckpointMismatch(
                f"{source}: {name} has shape {tuple(state[name].shape)} in the checkpoint but "
                f"{tuple(params[name].shape)} in the model; the checkpoint was trained with another model "
                f"configuration (for example model.lora_r). Use the config.yaml of the run that wrote it."
            )
    model.load_trainable_state_dict(state, strict=False)
    return len(loaded)


def load_finished_stages(model, run_dir: Path, stage_names: Sequence[str],
                         upto: Optional[str] = None) -> Tuple[List[str], Set[str]]:
    """Load ``final.pt`` of the finished stages, in order, up to and including ``upto`` (default: all).

    A stage's file holds the tensors that were trainable in that stage; a later stage may train more of them
    and overwrites earlier values, so the stages are applied one after the other. A pruned stage is skipped:
    a later stage's file holds all of its tensors. Returns the names of the stages whose files were loaded and
    the names of all tensors that were loaded.
    """
    loaded: List[str] = []
    tensors: Set[str] = set()
    for name in stage_names:
        final = find_final(run_dir, name)
        if final is not None:
            state = load_checkpoint(final)["model"]
            load_state_checked(model, state, str(final))
            loaded.append(name)
            tensors.update(state)
        elif not is_pruned(run_dir, name):
            break
        if name == upto:
            break
    return loaded, tensors


def prune_superseded(run_dir: Path, stage_names: Sequence[str]) -> List[str]:
    """Delete the ``final.pt`` of earlier stages whose tensors all reappear in the last stage's ``final.pt``.

    The last stage's weights are then complete; the earlier files only cost disk space (about 1 GB each for
    LLaVA-7B). A ``final.pt.pruned`` marker keeps the stage counted as finished. Returns the pruned stages.
    """
    run_dir = Path(run_dir)
    last = find_final(run_dir, stage_names[-1])
    if last is None:
        raise FileNotFoundError(f"cannot prune: the last stage {stage_names[-1]} has no {FINAL}")
    last_names = set(load_checkpoint(last)["model"])
    pruned: List[str] = []
    for name in stage_names[:-1]:
        final = find_final(run_dir, name)
        if final is None:
            continue
        names = set(load_checkpoint(final)["model"])
        if not names <= last_names:
            logger.info("[%s] kept: %d tensors are not in the last stage's checkpoint", name, len(names - last_names))
            continue
        write_json(run_dir / name / PRUNED, {"superseded_by": stage_names[-1], "tensors": len(names)})
        final.unlink()
        pruned.append(name)
    return pruned
