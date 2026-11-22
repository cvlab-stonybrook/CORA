"""Checks that run before the model is built, so that a mistake costs seconds instead of the stages before it.

* Stage order (paper Sec. 3.3): a stage needs the weights of the stages before it.
* Data: the labeled and unlabeled datasets of every stage that will run are constructed once and discarded, which
  surfaces missing instruction files, pseudo-labels and query databases at the start.
* Configuration: every finished stage (and every ``last.pt`` that would be resumed) stores a fingerprint of the
  settings that decide its weights. A run folder is not reused under changed settings unless that is allowed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cora import constants as C
from cora.config import Config, StageConfig
from cora.data.datasets import LabeledDataset, UnlabeledDataset
from cora.engine import checkpoint as ckpt
from cora.engine.evaluator import available_sets

# Settings that do not change the weights a stage produces.
_IGNORED = {
    "train": {"stages", "run_name", "log_every", "save_every", "eval_every", "resume"},
    "data": {"num_workers", "eval_ignore_void"},
}
# Data settings that only reach stages with unlabeled images / with the conditional set.
_UNLABELED_ONLY_DATA = {"ssl_method"}
_CONDITIONAL_ONLY_DATA = {"conditional_sources"}


def stage_total_steps(cfg: Config, stage: StageConfig) -> int:
    """Optimizer steps of a stage (Sec. 3.3: equal for all stages unless ``stage.iters`` is set)."""
    return stage.iters if stage.iters is not None else cfg.train.iters_per_stage


def _normalise(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def stage_fingerprint(cfg: Config, stage: StageConfig, world_size: int = 1) -> Dict[str, Any]:
    """The settings that decide the weights of ``stage`` given the weights of the stages before it.

    The unlabeled and token-alignment settings count only for stages that use unlabeled images, so that
    runs of one partition that differ only in these settings can share their first two stages.
    """
    flat = asdict(cfg)
    train = {k: v for k, v in flat["train"].items() if k not in _IGNORED["train"]}
    data = {k: v for k, v in flat["data"].items() if k not in _IGNORED["data"]}
    if not stage.use_unlabeled:  # e.g. the UniMatch and SemiVL runs of one partition share stages 1 and 2
        data = {k: v for k, v in data.items() if k not in _UNLABELED_ONLY_DATA}
    if C.SET_CONDITIONAL not in stage.labeled_sets:
        data = {k: v for k, v in data.items() if k not in _CONDITIONAL_ONLY_DATA}
    fingerprint: Dict[str, Any] = {
        "stage": asdict(stage),
        "steps": stage_total_steps(cfg, stage),
        "world_size": world_size,
        "data": data,
        "model": flat["model"],
        "loss": flat["loss"],
        "train": train,
    }
    if stage.use_unlabeled:
        fingerprint["unlabeled"] = flat["unlabeled"]
        fingerprint["tfca"] = flat["tfca"]
    return _normalise(fingerprint)


# Settings that were added after stages had been trained, with the value that reproduces the earlier behaviour: a
# stored fingerprint without the key is read as having that value, so those stages can still be continued and linked.
_ADDED_LATER: Dict[str, Any] = {
    "data.reasonseg_train_share": 0.0,
    "data.reasonseg_train_images": None,
    "model.text_hidden_fcs_init": None,
    "model.pretrained_seg_token": False,
    "model.train_seg_projection": True,
}


def diff_fingerprints(saved: Any, current: Any, prefix: str = "") -> List[str]:
    """Dotted keys at which two fingerprints differ, each as ``key: saved -> current``."""
    if isinstance(saved, dict) and isinstance(current, dict):
        out: List[str] = []
        for key in sorted(set(saved) | set(current)):
            dotted = f"{prefix}{key}"
            before = saved.get(key, _ADDED_LATER.get(dotted, "<absent>")) if dotted in _ADDED_LATER else saved.get(key, "<absent>")
            out.extend(diff_fingerprints(before, current.get(key, "<absent>"), f"{dotted}."))
        return out
    return [] if saved == current else [f"{prefix[:-1]}: {saved!r} -> {current!r}"]


@dataclass
class Plan:
    """What :func:`preflight` found."""

    to_run: List[str] = field(default_factory=list)  # stages that will be trained (from scratch or resumed)
    finished: List[str] = field(default_factory=list)  # stages that already have their weights
    adopt: Dict[str, Path] = field(default_factory=dict)  # stage -> final.pt of another run to link in
    eval_sets: List[str] = field(default_factory=list)  # validation sets that exist
    missing_sets: List[str] = field(default_factory=list)  # configured validation sets without instruction files


def resolve_only_stages(cfg: Config, only_stages: Optional[Sequence[str]]) -> List[str]:
    """Validate the stage names of the configuration and of a ``--stages`` selection."""
    names = [s.name for s in cfg.train.stages]
    if len(set(names)) != len(names):
        raise ValueError(f"stage names must be unique, got {names}")
    selected = list(only_stages) if only_stages else names
    unknown = [s for s in selected if s not in names]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; configured stages: {names}")
    return selected


def _stage_hint(cfg: Config, stage: StageConfig, kind: str) -> str:
    if kind == "unlabeled":
        return (f" -- the pseudo-labels of stage {stage.name} are read from {cfg.pseudo_dir}, chosen by "
                f"data.ssl_method={cfg.data.ssl_method!r} and data.partition={cfg.data.partition!r}; finish "
                f"tools/make_pseudo_labels.py for them, or set unlabeled.allow_partial=true to train on the "
                f"images that have a pseudo-label")
    return (f" -- stage {stage.name} needs the instruction sets {stage.labeled_sets} for the labeled images of "
            f"partition {cfg.data.partition!r} under {cfg.data_dir / 'instructions' / 'train'}")


def _check_datasets(cfg: Config, stage: StageConfig, spec, log: logging.Logger) -> None:
    """Construct (and discard) the datasets of a stage, adding the stage and a remedy to their errors."""
    for kind, build in (
        ("labeled", lambda: LabeledDataset(cfg, stage, spec, seed=cfg.train.seed)),
        ("unlabeled", lambda: UnlabeledDataset(cfg, spec, seed=cfg.train.seed + 1, allow_partial=cfg.unlabeled.allow_partial)),
    ):
        if kind == "unlabeled" and not stage.use_unlabeled:
            continue
        try:
            build()
        except (FileNotFoundError, ValueError) as exc:
            # the ReasonSeg folder has its own remedy in the message; the hint about the instruction sets would mislead
            hint = "" if "reasonseg_train_share" in str(exc) else _stage_hint(cfg, stage, kind)
            raise type(exc)(f"[{stage.name}] {kind} data: {exc}{hint}") from exc
    log.info(f"[{stage.name}] data checked")


def _check_fingerprint(path_dir: Path, cfg: Config, stage: StageConfig, world_size: int, what: str,
                       allow_change: bool, log: logging.Logger) -> None:
    saved = ckpt.read_json(path_dir / ckpt.FINGERPRINT)
    if saved is None:
        log.warning(f"[{stage.name}] {what} has no fingerprint.json (written by an older version); "
                    f"cannot verify that the settings are unchanged")
        return
    diffs = diff_fingerprints(saved, stage_fingerprint(cfg, stage, world_size))
    if not diffs:
        return
    listing = "\n  ".join(diffs[:20]) + ("\n  ..." if len(diffs) > 20 else "")
    message = (
        f"[{stage.name}] the settings differ from those that produced the existing {what} in {path_dir}:\n  {listing}\n"
        f"Use another train.run_name for the new settings (tools/train.py --init-stages-from <run_dir> reuses the "
        f"stages that do not depend on them), or pass --allow-config-change to continue with the existing "
        f"{what} anyway."
    )
    if not allow_change:
        raise ValueError(message)
    log.warning(message.split("\nUse another")[0] + "\n(continuing: --allow-config-change)")


def find_adoptable(cfg: Config, stage: StageConfig, source: Path, world_size: int) -> Optional[Path]:
    """``final.pt`` of ``stage`` in the run folder ``source`` when it was trained with identical settings."""
    final = ckpt.find_final(source, stage.name)
    if final is None:
        return None
    saved = ckpt.read_json(Path(source) / stage.name / ckpt.FINGERPRINT)
    if saved is None or diff_fingerprints(saved, stage_fingerprint(cfg, stage, world_size)):
        return None
    return final


def preflight(
    cfg: Config,
    spec,
    only_stages: Optional[Sequence[str]] = None,
    *,
    world_size: int = 1,
    allow_config_change: bool = False,
    init_stages_from: Optional[Path] = None,
    log: Optional[logging.Logger] = None,
) -> Plan:
    """Check everything that can be checked without the model; raise at once with a remedy on a problem.

    Raises ``ValueError`` for unknown stage names, changed settings or ``train.resume=false`` on a finished
    stage, ``RuntimeError`` when a stage that is needed but not selected has no weights, and the datasets'
    ``FileNotFoundError`` / ``ValueError`` (with the stage and a remedy added) for missing or inconsistent data.
    """
    log = log or logging.getLogger("cora")
    selected = resolve_only_stages(cfg, only_stages)
    stages = cfg.train.stages
    last_wanted = max(i for i, s in enumerate(stages) if s.name in selected)
    run_dir = cfg.run_dir
    plan = Plan()
    chain_intact = True  # a stage trained anew makes the weights of the stages after it differ from another run's
    missing: List[str] = []
    for stage in stages[: last_wanted + 1]:
        stage_dir = run_dir / stage.name
        if ckpt.is_finished(run_dir, stage.name):
            if stage.name in selected and not cfg.train.resume:
                raise ValueError(
                    f"[{stage.name}] already finished in {run_dir} and train.resume=false: use another "
                    f"train.run_name, or set train.resume=true to reuse the finished stage"
                )
            if ckpt.find_final(run_dir, stage.name) is not None:
                _check_fingerprint(stage_dir, cfg, stage, world_size, "final.pt", allow_config_change, log)
            plan.finished.append(stage.name)
            continue
        adopted = find_adoptable(cfg, stage, Path(init_stages_from), world_size) if init_stages_from and chain_intact else None
        if adopted is not None:
            plan.adopt[stage.name] = adopted
            plan.finished.append(stage.name)
            continue
        chain_intact = False
        if stage.name not in selected:
            missing.append(stage.name)
            continue
        if cfg.train.resume and ckpt.find_last(run_dir, stage.name) is not None:
            _check_fingerprint(stage_dir, cfg, stage, world_size, "last.pt", allow_config_change, log)
        plan.to_run.append(stage.name)

    if missing:
        selection = ",".join(missing + [s for s in selected if s not in missing])
        raise RuntimeError(
            f"stage(s) {missing} must be trained before {selected}: no final.pt in {run_dir}. "
            f"Run `python tools/train.py --stages {selection}` (add your --config and overrides), "
            f"or point --init-stages-from at a run folder that finished them with the same settings."
        )

    for stage in stages:
        if stage.name in plan.to_run:
            _check_datasets(cfg, stage, spec, log)

    plan.eval_sets, plan.missing_sets = available_sets(cfg, "val", cfg.eval.sets)
    if plan.missing_sets:
        log.info(f"no validation instructions for {plan.missing_sets}: they are skipped "
                 f"(expected under {cfg.data_dir / 'instructions' / 'val'})")
    if not plan.eval_sets:
        log.warning(f"none of the validation sets {cfg.eval.sets} exists: the run will end without any validation")
    return plan
