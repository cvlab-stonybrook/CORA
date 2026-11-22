"""Three-stage CORA training (paper Sec. 3.3) with the objective of Eq. 9.

Stage 1: semantic and attribute-based instructions on labeled images.
Stage 2: conditional (spatial) visual instructions on labeled images, with the anchor encoders.
Stage 3: labeled images plus unlabeled images whose pseudo-labels are weighted per pixel by output
         consistency (Eq. 6, Eq. 7) and whose <SSEG> tokens are aligned with labeled ones (Eq. 8):
         L = L^l + sigma * (L^u_seg + L^u_t)                                             (Eq. 9)
Every stage runs for the same number of optimizer steps and continues from the previous stage's weights.

A stage resumed from ``last.pt`` continues exactly where the interrupted run stopped: the model, optimizer,
scheduler, token bank, teacher, the position in both data streams and the random generators of every rank are
restored (with ``lora_dropout=0`` the result equals an uninterrupted run).
"""
from __future__ import annotations

import contextlib
import gc
import json
import math
import signal
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from cora import constants as C
from cora.config import Config, StageConfig, save_config, to_dict
from cora.data.collate import make_collate
from cora.data.datasets import EvalDataset, LabeledDataset, UnlabeledDataset
from cora.engine import checkpoint as ckpt
from cora.engine import dist as D
from cora.engine.evaluator import available_sets, evaluate, format_results, shard_indices
from cora.engine.logs import MetricWriter, setup_logger
from cora.engine.optim import build_optimizer, build_scheduler
from cora.engine.preflight import Plan, preflight, resolve_only_stages, stage_fingerprint, stage_total_steps
from cora.engine.streams import BatchStream
from cora.losses.contrastive import token_contrastive_loss
from cora.losses.criterion import labeled_loss, pixel_weights, unlabeled_loss
from cora.losses.token_bank import TokenBank
from cora.model.build import build_model, build_tokenizer
from cora.model.teacher import Teacher

PAPER_BATCH_SIZE = 8  # Sec. 4.1: batch size 8 (2 per device on 4 GPUs)
MAX_CONSECUTIVE_SKIPPED_STEPS = 5  # optimizer steps with a non-finite gradient tolerated in a row
UNLABELED_RATIO_WARNING = 50.0  # warn when sigma * L^u is this many times the labeled loss (Eq. 9)


class TrainingInterrupted(RuntimeError):
    """Training stopped on SIGINT / SIGTERM after saving ``last.pt``; running the same command resumes it.

    ``exit_code`` is the conventional shell status of the signal (130 for SIGINT, 143 for SIGTERM).
    """

    def __init__(self, message: str, exit_code: int = 130) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class _Averager:
    """Running means of logged scalars between two log lines."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    def add(self, values: Dict[str, float]) -> None:
        for key, value in values.items():
            self.sums[key] += float(value)
            self.counts[key] += 1

    def pop(self) -> Dict[str, float]:
        out = {key: self.sums[key] / self.counts[key] for key in self.sums}
        self.sums.clear()
        self.counts.clear()
        return out


class _StopRequest:
    """Turns SIGINT / SIGTERM into a flag that the step loop reads once per step.

    The loop then saves ``last.pt`` between two optimizer steps, when model, optimizer and data position agree.
    A second signal restores the previous handlers and interrupts at once.
    """

    def __init__(self) -> None:
        self.signal_name: Optional[str] = None
        self._previous: Dict[int, object] = {}

    def __enter__(self) -> "_StopRequest":
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def _restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous = {}

    def _handle(self, signum, frame) -> None:
        if self.signal_name is not None:
            self._restore()
            raise KeyboardInterrupt
        self.signal_name = signal.Signals(signum).name

    def __exit__(self, *exc) -> None:
        self._restore()


def _scalars(losses: Dict[str, torch.Tensor], prefix: str) -> Dict[str, float]:
    out = {}
    for key, value in losses.items():
        out[prefix + key] = float(value.detach().item()) if torch.is_tensor(value) else float(value)
    return out


def _params_finite(params: Sequence[torch.nn.Parameter]) -> bool:
    return bool(torch.stack([torch.isfinite(p.detach()).all() for p in params]).all().item())


class Trainer:
    """Runs the stages of ``cfg.train.stages``.

    ``only_stages`` selects the stages to train (earlier ones must be finished). ``allow_config_change`` lets a
    run folder be reused although the settings differ from those that produced its checkpoints.
    ``init_stages_from`` is another run folder whose finished stages, trained with identical settings, are linked
    in instead of being trained again. ``prune_checkpoints`` deletes the ``final.pt`` of earlier stages once the
    last stage has finished and contains all of their tensors.
    """

    def __init__(
        self,
        cfg: Config,
        only_stages: Optional[Sequence[str]] = None,
        *,
        allow_config_change: bool = False,
        init_stages_from: Optional[Path] = None,
        prune_checkpoints: bool = False,
    ) -> None:
        self.cfg = cfg
        self.distributed, self.rank, self.world_size, self.device = D.init_distributed()
        self.main = self.rank == 0
        D.seed_everything(cfg.train.seed + self.rank)
        self.run_dir: Path = cfg.run_dir
        self.log = setup_logger(self.run_dir, self.main)
        self.metrics = MetricWriter(self.run_dir, enabled=self.main)
        self.allow_config_change = allow_config_change
        self.prune_checkpoints = prune_checkpoints
        self.only_stages = resolve_only_stages(cfg, only_stages)
        self.tokenizer, self.spec = build_tokenizer(cfg)
        # Everything that can be checked without the model is checked first (see cora.engine.preflight).
        self.plan: Plan = preflight(
            cfg, self.spec, self.only_stages, world_size=self.world_size,
            allow_config_change=allow_config_change, init_stages_from=init_stages_from, log=self.log,
        )
        if self.main:
            self._record_config()
            self._adopt_stages()
        D.barrier()
        self.model = build_model(cfg, self.spec).to(self.device)
        self.collate = make_collate(self.tokenizer, self.spec, cfg)
        self.eval_collate = make_collate(self.tokenizer, self.spec, cfg, generation=cfg.eval.mode == "generate")
        self.log.info(
            f"run {cfg.train.run_name}: dataset {cfg.data.dataset}, partition {cfg.data.partition}, "
            f"pseudo-labels {cfg.data.ssl_method}, {self.world_size} process(es), device {self.device}"
        )

    # ------------------------------------------------------------------------------------------
    def _record_config(self) -> None:
        """``config.yaml`` keeps the first configuration; a later invocation that differs adds ``config.<time>.yaml``."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(self.run_dir.glob("config*.yaml"), key=lambda p: p.stat().st_mtime)
        current = json.loads(json.dumps(to_dict(self.cfg), default=str))
        if files:
            with open(files[-1]) as f:
                if json.loads(json.dumps(yaml.safe_load(f), default=str)) == current:
                    return
            save_config(self.cfg, self.run_dir / f"config.{time.strftime('%Y%m%d-%H%M%S')}.yaml")
        else:
            save_config(self.cfg, self.run_dir / "config.yaml")

    def _adopt_stages(self) -> None:
        """Link in the finished stages of another run (``--init-stages-from``) that the preflight accepted."""
        for stage in self.cfg.train.stages:
            source = self.plan.adopt.get(stage.name)
            if source is None:
                continue
            stage_dir = self.run_dir / stage.name
            stage_dir.mkdir(parents=True, exist_ok=True)
            link = stage_dir / ckpt.FINAL
            link.unlink(missing_ok=True)
            link.symlink_to(Path(source).resolve())
            for name in (ckpt.FINGERPRINT, ckpt.METRICS):
                content = ckpt.read_json(Path(source).parent / name)
                if content is not None:
                    ckpt.write_json(stage_dir / name, content)
            self.log.info(f"[{stage.name}] linked from {source}")

    # ------------------------------------------------------------------------------------------
    def fit(self) -> Dict[str, dict]:
        """Run the configured stages in order; a finished stage is skipped (its weights are loaded)."""
        stages = self.cfg.train.stages
        names = [s.name for s in stages]
        last_wanted = max(i for i, s in enumerate(stages) if s.name in self.only_stages)
        results: Dict[str, dict] = {}
        for stage in stages[: last_wanted + 1]:
            if ckpt.is_finished(self.run_dir, stage.name):
                final = ckpt.find_final(self.run_dir, stage.name)
                if final is None:
                    self.log.info(f"[{stage.name}] already trained; its checkpoint was pruned")
                    continue
                self._load_weights(final)
                self.log.info(f"[{stage.name}] already trained; weights loaded from {final}")
                if stage.name in self.only_stages and ckpt.read_json(self.run_dir / stage.name / ckpt.METRICS) is None:
                    # The run ended between final.pt and the stage evaluation: do the evaluation now.
                    self.log.info(f"[{stage.name}] no metrics.json: evaluating the stored weights")
                    results[stage.name] = self._finish_stage(stage)
                continue
            results[stage.name] = self.run_stage(stage)
        if self.prune_checkpoints and last_wanted == len(stages) - 1 and self.main \
                and ckpt.find_final(self.run_dir, names[-1]) is not None:
            pruned = ckpt.prune_superseded(self.run_dir, names)
            if pruned:
                self.log.info(f"pruned the checkpoints of {pruned}; {names[-1]}/final.pt holds all their tensors")
        self.metrics.close()
        return results

    def _load_weights(self, path: Path) -> None:
        # A stage's checkpoint holds the tensors that were trainable in that stage; later stages may train more.
        ckpt.load_state_checked(self.model, ckpt.load_checkpoint(path)["model"], str(path))

    # ------------------------------------------------------------------------------------------
    def _sync_context(self, net, last_backward: bool):
        """Gradient all-reduce only on the last backward pass of an optimizer step (DDP ``no_sync`` otherwise)."""
        if self.distributed and not last_backward:
            return net.no_sync()
        return contextlib.nullcontext()

    def _enqueue_bank(self, bank: TokenBank, feats: torch.Tensor, class_ids: torch.Tensor) -> None:
        """Add the labeled <SSEG> features of this step to the bank (Sec. 3.2.2).

        With several processes the features of all ranks are gathered first, so that every rank holds the same
        single bank, as in the single-process method. Non-finite features are not stored.
        """
        feats = feats.detach().float()
        if self.distributed:
            parts = D.gather_objects((feats.cpu(), class_ids.cpu()))
            feats = torch.cat([p[0] for p in parts]).to(self.device)
            class_ids = torch.cat([p[1] for p in parts]).to(self.device)
        finite = torch.isfinite(feats).all(dim=-1)
        if not bool(finite.all()):
            feats, class_ids = feats[finite], class_ids[finite]
        bank.enqueue(feats, class_ids)

    def _check_processes_agree(self, stage: StageConfig, params: Sequence[torch.nn.Parameter]) -> None:
        """Every process must hold the same weights: they start equal and apply the same averaged gradient.

        Only the main process writes checkpoints, so weights that drift apart between the processes (a gradient
        that was not synchronised) would go unnoticed. A checksum of the trainable tensors is compared at the end
        of every stage; a difference stops the run before ``final.pt`` is written.
        """
        if not self.distributed:
            return
        checksum = float(torch.stack([p.detach().double().abs().sum() for p in params]).sum().item())
        sums = D.gather_objects(checksum)
        if max(sums) - min(sums) > 1e-7 * max(1.0, abs(max(sums))):
            raise RuntimeError(
                f"[{stage.name}] the {self.world_size} processes hold different weights (checksums {sums}): a gradient "
                f"was not synchronised between them. final.pt is not written."
            )
        self.log.info(f"[{stage.name}] the {self.world_size} processes hold identical weights")

    def _save_last(self, stage: StageConfig, stage_dir: Path, done: int, optimizer, scheduler, bank, teacher,
                   labeled: BatchStream, unlabeled: Optional[BatchStream], params, required: bool = True) -> bool:
        """Write ``last.pt`` (collective: every rank gathers its RNG state; only the main rank writes).

        ``required=False`` is for the periodic saves: when the file cannot be written (a full disk), training goes
        on and the previous ``last.pt`` stays, instead of losing the hours already spent on the stage. Returns
        whether the file was written (the same answer on every rank). A failure that stops the run (a required
        save, or an error that is not about writing the file) is raised on every rank, after all of them have
        learned about it: a rank that left alone would leave the others waiting in their next collective call.
        """
        if not _params_finite(params):
            raise FloatingPointError(
                f"[{stage.name}] the weights contain NaN or inf at step {done}; last.pt is not overwritten "
                f"(the last good state stays). Lower train.lr or train.grad_clip."
            )
        rng = D.gather_objects(D.capture_rng_state(self.device))
        error: Optional[Exception] = None
        if self.main:
            meta = {"stage": stage.name, "step": done, "labeled_batches": labeled.consumed,
                    "unlabeled_batches": unlabeled.consumed if unlabeled is not None else 0,
                    "world_size": self.world_size}
            try:
                ckpt.save_last(stage_dir / ckpt.LAST, self.model, optimizer, scheduler, bank, teacher, meta, rng)
            except Exception as caught:  # noqa: BLE001 - raised or logged below, once every rank knows
                error = caught
        if not D.any_rank(error is not None):
            return True
        # torch.save reports a failed write (a full disk) as OSError or RuntimeError; only those are tolerated.
        fatal = D.any_rank(error is not None and (required or not isinstance(error, (OSError, RuntimeError))))
        if fatal:
            if error is not None:
                raise error
            raise RuntimeError(f"[{stage.name}] the main process could not write {ckpt.LAST} at step {done} (see its error)")
        if error is not None:
            previous = "the previous one stays" if (stage_dir / ckpt.LAST).is_file() else "there is no earlier one"
            self.log.warning(
                f"[{stage.name}] could not write {ckpt.LAST} at step {done} ({type(error).__name__}: {error}); "
                f"training continues without this resumable state ({previous}). Check the free disk space."
            )
        return False

    def run_stage(self, stage: StageConfig) -> dict:
        cfg, log, model, device = self.cfg, self.log, self.model, self.device
        total = stage_total_steps(cfg, stage)
        stage_dir = self.run_dir / stage.name
        accum = max(1, cfg.train.grad_accum_steps)
        effective_batch = cfg.train.batch_size * self.world_size * accum
        if effective_batch != PAPER_BATCH_SIZE:
            log.warning(
                f"[{stage.name}] effective labeled batch size is {effective_batch} "
                f"(train.batch_size {cfg.train.batch_size} x {self.world_size} process(es) x train.grad_accum_steps "
                f"{accum}); the paper uses {PAPER_BATCH_SIZE} (Sec. 4.1). With this number of processes "
                f"train.grad_accum_steps={max(1, PAPER_BATCH_SIZE // max(1, cfg.train.batch_size * self.world_size))} "
                f"reaches it."
            )
        if self.main:
            stage_dir.mkdir(parents=True, exist_ok=True)
            for stale in ckpt.clean_stale_tmp(stage_dir):
                log.warning(f"[{stage.name}] removed the unfinished file {stale}")
            fingerprint = stage_fingerprint(cfg, stage, self.world_size)
            if ckpt.read_json(stage_dir / ckpt.FINGERPRINT) != fingerprint:
                ckpt.write_json(stage_dir / ckpt.FINGERPRINT, fingerprint)

        model.set_reference_trainable(stage.use_reference)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = build_optimizer(params, cfg.train)
        scheduler = build_scheduler(optimizer, cfg.train, total)

        # ---- resume: model, optimizer, scheduler, token bank, teacher, data position (RNG below) --------
        start = 0
        state = None
        last = ckpt.find_last(self.run_dir, stage.name)
        if last is not None and not cfg.train.resume:
            log.warning(f"[{stage.name}] train.resume=false: {last} is ignored and will be overwritten")
        elif last is not None:
            state = ckpt.load_checkpoint(last)
            start = int(state["meta"]["step"])
            if start >= total:
                raise ValueError(
                    f"[{stage.name}] {last} is at step {start}, but the stage has only {total} steps "
                    f"(train.iters_per_stage / stage.iters). Raise the step count or delete {last}."
                )
            model.load_trainable_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            saved_lr = optimizer.param_groups[0].get("initial_lr")
            if saved_lr is not None and not math.isclose(saved_lr, cfg.train.lr, rel_tol=1e-9):
                log.warning(f"[{stage.name}] learning rate, betas and weight decay come from {last} "
                            f"(lr {saved_lr:g}); train.lr={cfg.train.lr:g} is not applied to the resumed stage")
            dropped = self.metrics.drop_after(stage.name, start)
            if dropped:
                log.info(f"[{stage.name}] removed {dropped} metric rows after step {start}; they are written again")
            log.info(f"[{stage.name}] resumed from {last} at step {start}")
        if state is None:
            # The stage starts from its first step. Rows of an earlier attempt that ended without a usable last.pt
            # (a failed final.pt, train.resume=false) would otherwise stay next to the rows written now.
            dropped = self.metrics.drop_after(stage.name, -1)
            if dropped:
                log.info(f"[{stage.name}] removed {dropped} metric rows of an earlier attempt of this stage")
        meta = state["meta"] if state is not None else {}
        skip_labeled = int(meta.get("labeled_batches", start * accum))
        skip_unlabeled = int(meta.get("unlabeled_batches", start * accum))

        # ---- data streams: the same seed on every rank, so that the ranks read disjoint shares of one pass ----
        seed = cfg.train.seed
        labeled_stream = BatchStream(
            LabeledDataset(cfg, stage, self.spec, seed=seed), cfg.train.batch_size, self.collate,
            cfg.data.num_workers, seed=seed, distributed=self.distributed, skip_batches=skip_labeled,
        )
        unlabeled_stream: Optional[BatchStream] = None
        teacher: Optional[Teacher] = None
        bank: Optional[TokenBank] = None
        bank_state = state.get("bank") if state is not None else None
        use_bank = stage.use_unlabeled and cfg.tfca.enabled
        if bank_state is not None and not use_bank:
            log.warning(f"[{stage.name}] the token bank saved in {last} is discarded (tfca.enabled=false)")
        if stage.use_unlabeled:
            unlabeled_stream = BatchStream(
                UnlabeledDataset(cfg, self.spec, seed=seed + 1, allow_partial=cfg.unlabeled.allow_partial),
                cfg.unlabeled.batch_size, self.collate, cfg.data.num_workers, seed=seed + 1,
                distributed=self.distributed, skip_batches=skip_unlabeled,
            )
            if cfg.unlabeled.weighting == "consistency":
                # The K-1 rephrasings go through a network that they do not train (Fig. 3).
                teacher = Teacher(model, cfg.unlabeled.teacher, cfg.unlabeled.ema_momentum)
                teacher.start()
                if state is not None and "teacher" in state:
                    try:
                        teacher.load_state_dict(state["teacher"])
                    except ValueError as exc:
                        raise ValueError(
                            f"{exc}; set unlabeled.teacher to the value of the interrupted run, or use a new train.run_name"
                        ) from exc

        net = model
        if self.distributed:
            # Not every trainable tensor takes part in every pass (e.g. lm_head when the unlabeled pass
            # has no text loss), hence find_unused_parameters.
            net = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                          find_unused_parameters=True)

        n_train = sum(p.numel() for p in params)
        log.info(
            f"[{stage.name}] {total} steps, labeled sets {stage.labeled_sets}, reference encoders "
            f"{'on' if stage.use_reference else 'off'}, unlabeled {'on' if stage.use_unlabeled else 'off'}, "
            f"{n_train / 1e6:.1f} M trainable parameters, batch {cfg.train.batch_size} x {self.world_size} "
            f"x accumulation {accum}"
        )

        # The generators continue from the saved state (set last: the set-up above draws random numbers).
        if state is not None:
            saved_rng = state.get("rng")
            if saved_rng is not None and len(saved_rng) == self.world_size:
                D.restore_rng_state(saved_rng[self.rank], device)
            else:
                log.warning(f"[{stage.name}] {last} has no random-generator state for {self.world_size} process(es); "
                            f"the resumed run is not bit-for-bit equal to an uninterrupted one")

        clip = cfg.train.grad_clip if cfg.train.grad_clip > 0 else float("inf")
        averager = _Averager()
        tic = time.time()
        steps_since_log = 0
        skipped_total = 0
        skipped_in_row = 0
        ratio_warned = False
        with _StopRequest() as stop:
            for step in range(start, total):
                optimizer.zero_grad(set_to_none=True)
                for micro in range(accum):
                    # ---- labeled images: L^l = alpha * L_ce + BCE + DICE (Eq. 4, Eq. 5) -----------------
                    batch = next(labeled_stream).to(device)
                    last_backward = micro == accum - 1 and unlabeled_stream is None
                    with self._sync_context(net, last_backward):
                        out = net(batch)
                        losses = labeled_loss(out, batch, cfg.loss)
                        loss = losses["loss"]
                        if use_bank:
                            # Token bank of <SSEG> features of labeled images (Sec. 3.2.2).
                            feats = out.seg_embed if cfg.tfca.feature == "projected" else out.seg_hidden
                            if bank is None:
                                bank = TokenBank(len(C.class_names(cfg.data.dataset)), feats.shape[-1], cfg.tfca.bank_size).to(device)
                                if bank_state is not None:
                                    bank.load_state_dict(bank_state)
                            if cfg.tfca.labeled_anchor:
                                # Optional: labeled tokens repel bank tokens of other classes too (drawn in Fig. 4).
                                bank_features, bank_classes = bank.get()
                                extra, _ = token_contrastive_loss(
                                    feats, batch.class_ids, bank_features, bank_classes, cfg.tfca.temperature,
                                    cfg.tfca.num_positives, cfg.tfca.num_negatives, cfg.tfca.reduction)
                                loss = loss + cfg.unlabeled.weight * extra
                                losses["tfca_labeled"] = extra
                            self._enqueue_bank(bank, feats, batch.class_ids)
                        (loss / accum).backward()
                    averager.add(_scalars(losses, "labeled/"))
                    del out, losses, loss, batch

                    # ---- unlabeled images: sigma * (L^u_seg + L^u_t) (Eq. 6 to Eq. 9) -------------------
                    if unlabeled_stream is not None:
                        ubatch = next(unlabeled_stream).to(device)
                        sam = model.encode_sam(ubatch.images_sam)
                        teacher_logits = teacher.predict(ubatch.aux, sam) if teacher is not None else None
                        weights = pixel_weights(ubatch, teacher_logits, cfg.unlabeled)
                        del teacher_logits
                        with self._sync_context(net, micro == accum - 1):
                            uout = net(ubatch, compute_text_loss=cfg.unlabeled.text_loss, sam_embeddings=sam)
                            ulosses = unlabeled_loss(uout, ubatch, weights, bank, cfg.unlabeled, cfg.tfca)
                            uloss = ulosses["loss"]
                            if cfg.unlabeled.text_loss and uout.ce_loss is not None:
                                uloss = uloss + cfg.unlabeled.weight * cfg.loss.ce_weight * uout.ce_loss
                            (uloss / accum).backward()
                        averager.add(_scalars(ulosses, "unlabeled/"))
                        del uout, ulosses, uloss, weights, ubatch, sam

                grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
                lr_used = optimizer.param_groups[0]["lr"]  # the rate this step applies
                if bool(torch.isfinite(grad_norm)):
                    optimizer.step()
                    if teacher is not None:
                        teacher.update()
                    averager.add({"grad_norm": float(grad_norm)})
                    skipped_in_row = 0
                else:
                    # The gradient is the same on every rank after the all-reduce, so all ranks skip together.
                    skipped_total += 1
                    skipped_in_row += 1
                    log.warning(f"[{stage.name}] step {step + 1}: non-finite gradient norm, optimizer step skipped "
                                f"({skipped_in_row} in a row, {skipped_total} in this stage)")
                    if skipped_in_row >= MAX_CONSECUTIVE_SKIPPED_STEPS:
                        raise FloatingPointError(
                            f"[{stage.name}] {skipped_in_row} consecutive steps with a non-finite gradient (up to step "
                            f"{step + 1}); stopping before the weights are saved. Lower train.lr or train.grad_clip, "
                            f"or inspect the losses in {self.run_dir / 'train.log'}."
                        )
                scheduler.step()
                steps_since_log += 1

                done = step + 1
                if self.main and (done % cfg.train.log_every == 0 or done == total):
                    scalars = averager.pop()
                    elapsed = time.time() - tic
                    rate = steps_since_log / max(elapsed, 1e-9)
                    scalars["lr"] = lr_used
                    scalars["steps_per_s"] = rate
                    if skipped_total:
                        scalars["skipped_steps"] = float(skipped_total)
                    if "unlabeled/loss" in scalars and scalars.get("labeled/loss"):
                        # How large the sigma-scaled unlabeled term is next to the labeled loss.
                        scalars["unlabeled_to_labeled"] = scalars["unlabeled/loss"] / scalars["labeled/loss"]
                        if scalars["unlabeled_to_labeled"] > UNLABELED_RATIO_WARNING and not ratio_warned:
                            ratio_warned = True
                            log.warning(
                                f"[{stage.name}] sigma * L^u is {scalars['unlabeled_to_labeled']:.0f} times the labeled "
                                f"loss: with unlabeled.reduction={cfg.unlabeled.reduction!r} the sum over all pixels "
                                f"outweighs the labeled terms at unlabeled.weight={cfg.unlabeled.weight:g}. This follows "
                                f"Eq. 7 and Eq. 9 literally; unlabeled.reduction=mean or a smaller unlabeled.weight "
                                f"restores the balance."
                            )
                    self.metrics.write(stage.name, done, scalars)
                    eta_min = (total - done) / max(rate, 1e-9) / 60.0
                    shown = " ".join(f"{k}={v:.4g}" for k, v in sorted(scalars.items()))
                    log.info(f"[{stage.name}] step {done}/{total} {shown} eta={eta_min:.0f}min")
                    tic = time.time()
                    steps_since_log = 0
                elif not self.main and done % cfg.train.log_every == 0:
                    averager.pop()

                saved_here = False
                if cfg.train.save_every > 0 and done % cfg.train.save_every == 0 and done < total:
                    saved_here = self._save_last(stage, stage_dir, done, optimizer, scheduler, bank, teacher,
                                                 labeled_stream, unlabeled_stream, params, required=False)
                if cfg.train.eval_every > 0 and done % cfg.train.eval_every == 0 and done < total:
                    optimizer.zero_grad(set_to_none=True)  # the gradients are not needed during evaluation
                    self.evaluate(stage, done, final=False)
                if D.any_rank(stop.signal_name is not None) and done < total:
                    if not saved_here:
                        self._save_last(stage, stage_dir, done, optimizer, scheduler, bank, teacher,
                                        labeled_stream, unlabeled_stream, params)
                    raise TrainingInterrupted(
                        f"[{stage.name}] stopped by {stop.signal_name or 'a signal on another process'} after step "
                        f"{done}/{total}; the state is saved in {stage_dir / ckpt.LAST}. Run the same command again to resume.",
                        exit_code=143 if stop.signal_name == "SIGTERM" else 130,
                    )

        if not _params_finite(params):
            raise FloatingPointError(f"[{stage.name}] the weights contain NaN or inf after step {total}; final.pt is not written")
        self._check_processes_agree(stage, params)
        final = stage_dir / ckpt.FINAL
        error: Optional[Exception] = None
        if self.main:
            try:
                ckpt.save_final(final, model, {"stage": stage.name, "step": total})
            except Exception as caught:  # noqa: BLE001 - raised below, once every rank knows
                error = caught
        # As in _save_last: no rank may leave before the others know, or they would wait in the next collective call.
        if D.any_rank(error is not None):
            if error is not None:
                raise error
            raise RuntimeError(f"[{stage.name}] the main process could not write {ckpt.FINAL} (see its error); "
                               f"a {ckpt.LAST} of this stage, if there is one, still resumes it")
        if self.main:
            (stage_dir / ckpt.LAST).unlink(missing_ok=True)
            log.info(f"[{stage.name}] finished; weights saved to {final} ({final.stat().st_size / 2**20:.0f} MiB)")
        D.barrier()
        # Release the loader workers (and their prefetched batches), the DDP buckets, the gradients and the
        # teacher snapshot before the evaluation starts its own workers.
        labeled_stream.close()
        if unlabeled_stream is not None:
            unlabeled_stream.close()
        optimizer.zero_grad(set_to_none=True)
        del net, teacher, bank, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self._finish_stage(stage)

    def _finish_stage(self, stage: StageConfig) -> dict:
        """Evaluate the stage's weights and write ``metrics.json`` (an empty file when no validation set exists)."""
        results = self.evaluate(stage, stage_total_steps(self.cfg, stage), final=True)
        if self.main:
            ckpt.write_json(self.run_dir / stage.name / ckpt.METRICS, results)
        return results

    # ------------------------------------------------------------------------------------------
    def evaluate(self, stage: StageConfig, step: int, final: bool) -> dict:
        """cIoU / gIoU on the validation instruction sets that exist on disk (teacher-forced by default).

        The random generators are left as they were found (a data loader draws its seed from the global generator):
        an evaluation between two training steps must not change what the following steps draw, or a run resumed
        from the ``last.pt`` written in that step would not continue like the uninterrupted one.
        """
        rng = D.capture_rng_state(self.device)
        try:
            return self._evaluate(stage, step, final)
        finally:
            D.restore_rng_state(rng, self.device)

    def _evaluate(self, stage: StageConfig, step: int, final: bool) -> dict:
        cfg = self.cfg
        tag = stage.name if final else f"{stage.name}@{step}"
        sets, missing = available_sets(cfg, "val", cfg.eval.sets)
        if missing:
            self.log.info(f"[{tag}] no validation instructions for {missing}; skipped")
        if not sets:
            return {}
        dataset = EvalDataset(cfg, self.spec, sets, split="val")
        # Evaluations during training may use a smaller subset than tools/evaluate.py (eval.stage_max_samples).
        limit = cfg.eval.stage_max_samples if cfg.eval.stage_max_samples is not None else cfg.eval.max_samples
        indices = shard_indices(len(dataset), self.rank, self.world_size, limit)
        n_total = len(shard_indices(len(dataset), 0, 1, limit))
        self.log.info(f"[{tag}] evaluating {n_total} of {len(dataset)} validation samples ({cfg.eval.mode}, sets {sets})")
        loader = DataLoader(Subset(dataset, indices), batch_size=cfg.eval.batch_size, shuffle=False,
                            num_workers=min(4, cfg.data.num_workers), collate_fn=self.eval_collate)
        tic = time.time()
        results = evaluate(self.model, loader, self.device, cfg.eval.mode, cfg.eval.max_new_tokens)
        if self.main:
            self.log.info(f"[{tag}] validation ({cfg.eval.mode}, {time.time() - tic:.0f} s)\n{format_results(results)}")
            rows = {f"val/{g}/{key}": float(v[key]) for g, v in results.items() for key in ("ciou", "giou", "n")}
            self.metrics.write(stage.name, step, rows)
        return results
