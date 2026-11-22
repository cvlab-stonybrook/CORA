"""Tests of the engine parts that need no model: schedule, IoU accounting, checkpoints."""
import math
from types import SimpleNamespace

import pytest
import torch

from cora.config import TrainConfig
from cora.engine import checkpoint as ckpt
from cora.engine.evaluator import IoUMeter, evaluate, format_results, intersection_union, shard_indices
from cora.engine.optim import build_optimizer, build_scheduler, lr_factor


def test_lr_schedule_warmup_then_linear_decay():
    total, warmup = 1000, 100
    assert lr_factor(0, total, warmup, "warmup_linear_decay") == pytest.approx(1 / 100)
    assert lr_factor(99, total, warmup, "warmup_linear_decay") == pytest.approx(1.0)
    assert lr_factor(100, total, warmup, "warmup_linear_decay") == pytest.approx(1.0)
    assert lr_factor(550, total, warmup, "warmup_linear_decay") == pytest.approx(0.5)
    assert lr_factor(999, total, warmup, "warmup_linear_decay") == pytest.approx(1 / 900)
    assert lr_factor(550, total, warmup, "constant") == 1.0
    # monotone: up during warm-up, down afterwards
    values = [lr_factor(s, total, warmup, "warmup_linear_decay") for s in range(total)]
    assert all(b >= a for a, b in zip(values[:warmup], values[1:warmup]))
    assert all(b <= a for a, b in zip(values[warmup:], values[warmup + 1:]))


def test_lr_schedule_short_stage():
    # a stage shorter than the warm-up must still reach a positive rate
    assert all(lr_factor(s, 5, 100, "warmup_linear_decay") > 0 for s in range(5))


def test_optimizer_and_scheduler_follow_config():
    tcfg = TrainConfig(lr=2e-5, warmup_steps=2)
    w = torch.nn.Parameter(torch.zeros(3))
    frozen = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
    opt = build_optimizer([w, frozen], tcfg)
    assert len(opt.param_groups[0]["params"]) == 1
    assert opt.defaults["betas"] == (0.9, 0.95) and opt.defaults["weight_decay"] == 0.0
    sched = build_scheduler(opt, tcfg, total_steps=10)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-5 * 0.5)
    opt.step(); sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-5)
    with pytest.raises(ValueError):
        build_optimizer([frozen], tcfg)


def test_intersection_union_ignores_255():
    target = torch.tensor([[1, 1, 0, 255], [0, 0, 1, 255]], dtype=torch.uint8)
    pred = torch.tensor([[1, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    inter, union = intersection_union(pred, target)
    assert (inter, union) == (2.0, 4.0)
    with pytest.raises(ValueError):
        intersection_union(pred[:1], target)


def test_meter_ciou_giou_definitions():
    meter = IoUMeter()
    meter.update("all", 10, 20)   # IoU 0.5
    meter.update("all", 90, 100)  # IoU 0.9
    meter.update("all", 0, 0)     # empty target and empty prediction: counts as 1
    out = meter.compute()["all"]
    assert out["n"] == 3
    assert out["ciou"] == pytest.approx(100 * 100 / 120)
    assert out["giou"] == pytest.approx(100 * (0.5 + 0.9 + 1.0) / 3, abs=1e-3)
    other = IoUMeter()
    other.update("all", 20, 80)
    meter.merge(other.stats)
    assert meter.compute()["all"]["ciou"] == pytest.approx(100 * 120 / 200)
    assert "cIoU" in format_results(meter.compute())


def test_shard_indices_cover_the_dataset_once():
    shards = [shard_indices(10, rank, 3) for rank in range(3)]
    assert sorted(i for s in shards for i in s) == list(range(10))
    spread = shard_indices(100, 0, 1, max_samples=4)
    assert spread == [0, 25, 50, 75]  # evenly spread, not the first four
    assert sorted(i for r in range(2) for i in shard_indices(100, r, 2, max_samples=4)) == spread
    assert shard_indices(3, 0, 1, max_samples=10) == [0, 1, 2]


class _FakeModel(torch.nn.Module):
    """Predicts the target itself for even sequences and nothing for odd ones."""

    def forward(self, batch, compute_text_loss=True, sam_embeddings=None):
        logits = []
        for i, mask in enumerate(batch.masks):
            perfect = (mask == 1).float() * 2 - 1
            logits.append(perfect if i % 2 == 0 else -torch.ones_like(perfect))
        return SimpleNamespace(mask_logits=logits)

    def generate(self, batch, max_new_tokens=32):
        return ["" for _ in batch.masks], [None for _ in batch.masks]


class _FakeBatch(SimpleNamespace):
    def to(self, device):
        return self


def _batch():
    a = torch.zeros(4, 4, dtype=torch.uint8); a[:2] = 1          # 8 foreground pixels
    b = torch.zeros(4, 4, dtype=torch.uint8); b[0, 0] = 1; b[3] = 255
    return _FakeBatch(masks=[a, b], set_names=["conditional", "attribute"], is_long_query=[None, True])


def test_evaluate_groups_and_modes():
    model = _FakeModel()
    model.train()
    out = evaluate(model, [_batch()], torch.device("cpu"))
    assert model.training, "training mode must be restored"
    assert out["conditional"]["ciou"] == pytest.approx(100.0)
    assert out["attribute"]["ciou"] == pytest.approx(0.0)
    assert out["long"]["n"] == 1 and "short" not in out
    assert out["all"]["ciou"] == pytest.approx(100 * 8 / 9)
    generated = evaluate(model, [_batch()], torch.device("cpu"), mode="generate")
    assert generated["all"]["ciou"] == pytest.approx(0.0)  # no <SSEG> emitted counts as an empty prediction
    with pytest.raises(ValueError):
        evaluate(model, [], torch.device("cpu"), mode="bogus")


def test_checkpoint_round_trip_is_weights_only(tmp_path):
    state = {"model": {"w": torch.arange(4.0)}, "meta": {"stage": "stage1", "step": 7}}
    path = tmp_path / "run" / "stage1" / "final.pt"
    ckpt.atomic_save(state, path)
    loaded = ckpt.load_checkpoint(path)
    assert torch.equal(loaded["model"]["w"], state["model"]["w"]) and loaded["meta"]["step"] == 7
    assert not path.with_name("final.pt.tmp").exists()
    assert ckpt.find_final(tmp_path / "run", "stage1") == path
    assert ckpt.find_last(tmp_path / "run", "stage1") is None
    assert math.isclose(float(loaded["model"]["w"].sum()), 6.0)


# ----------------------------------------------------------------------------------------------
# Evaluation accounting, checkpoints, RNG state, metric rows, fingerprints, signals
# ----------------------------------------------------------------------------------------------
def test_generate_mode_counts_answers_without_a_mask():
    model = _FakeModel()
    out = evaluate(model, [_batch()], torch.device("cpu"), mode="generate")
    assert out["all"]["no_mask"] == 2 and out["conditional"]["no_mask"] == 1
    assert "no mask" in format_results(out)
    assert evaluate(model, [_batch()], torch.device("cpu"))["all"]["no_mask"] == 0
    assert "no mask" not in format_results(evaluate(model, [_batch()], torch.device("cpu")))


def test_available_sets_splits_present_and_missing(tmp_path):
    from cora.config import Config
    from cora.engine.evaluator import available_sets

    cfg = Config()
    cfg.paths.data_dir = str(tmp_path)
    folder = cfg.data_dir / "instructions" / "val"
    folder.mkdir(parents=True)
    (folder / "semantic.jsonl").write_text("")
    (folder / "conditional_templated.jsonl").write_text("")
    assert available_sets(cfg, "val", ["conditional", "attribute", "semantic"]) == (["conditional", "semantic"], ["attribute"])


class _Tiny(torch.nn.Module):
    """Stand-in with the two model methods the checkpoint helpers use."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(2))
        self.b = torch.nn.Parameter(torch.zeros(3))

    def load_trainable_state_dict(self, state, strict=True):
        with torch.no_grad():
            for name, value in state.items():
                dict(self.named_parameters())[name].copy_(value)


def test_load_state_checked_names_the_cause_of_a_mismatch():
    model = _Tiny()
    assert ckpt.load_state_checked(model, {"a": torch.ones(2)}, "x.pt") == 1 and model.a.sum() == 2
    with pytest.raises(ckpt.CheckpointMismatch, match="another model configuration"):
        ckpt.load_state_checked(model, {"a": torch.ones(5)}, "x.pt")
    with pytest.raises(ckpt.CheckpointMismatch, match="none of its 1 tensors"):
        ckpt.load_state_checked(model, {"zzz": torch.ones(1)}, "x.pt")


def test_finished_stages_load_in_order_and_pruning_keeps_the_last(tmp_path):
    names = ["s1", "s2", "s3"]
    ckpt.atomic_save({"model": {"a": torch.full((2,), 1.0)}}, tmp_path / "s1" / "final.pt")
    ckpt.atomic_save({"model": {"a": torch.full((2,), 2.0), "b": torch.full((3,), 2.0)}}, tmp_path / "s2" / "final.pt")
    ckpt.atomic_save({"model": {"a": torch.full((2,), 3.0), "b": torch.full((3,), 3.0)}}, tmp_path / "s3" / "final.pt")
    model = _Tiny()
    loaded, tensors = ckpt.load_finished_stages(model, tmp_path, names, upto="s2")
    assert loaded == ["s1", "s2"] and tensors == {"a", "b"} and float(model.a[0]) == 2.0
    assert ckpt.load_finished_stages(_Tiny(), tmp_path, names)[0] == names
    assert ckpt.prune_superseded(tmp_path, names) == ["s1", "s2"]
    assert ckpt.find_final(tmp_path, "s1") is None and ckpt.is_pruned(tmp_path, "s1") and ckpt.is_finished(tmp_path, "s2")
    model = _Tiny()
    assert ckpt.load_finished_stages(model, tmp_path, names)[0] == ["s3"] and float(model.b[0]) == 3.0
    # a stage whose tensors the last stage does not hold is kept
    ckpt.atomic_save({"model": {"c": torch.zeros(1)}}, tmp_path / "s0" / "final.pt")
    assert ckpt.prune_superseded(tmp_path, ["s0", "s3"]) == [] and ckpt.find_final(tmp_path, "s0") is not None
    with pytest.raises(FileNotFoundError):
        ckpt.prune_superseded(tmp_path, ["s3", "missing"])


def test_stale_tmp_files_are_removed(tmp_path):
    (tmp_path / "last.pt.tmp").write_bytes(b"x")
    assert [p.name for p in ckpt.clean_stale_tmp(tmp_path)] == ["last.pt.tmp"] and not list(tmp_path.glob("*.tmp"))


def test_rng_state_round_trips_through_a_weights_only_checkpoint(tmp_path):
    import random

    import numpy as np

    from cora.engine import dist as D

    device = torch.device("cpu")
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    ckpt.atomic_save({"rng": [D.capture_rng_state(device)]}, tmp_path / "rng.pt")
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    D.restore_rng_state(ckpt.load_checkpoint(tmp_path / "rng.pt")["rng"][0], device)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected
    assert D.any_rank(False) is False and D.any_rank(True) is True


def test_metric_rows_beyond_a_resumed_step_are_dropped(tmp_path):
    import json

    from cora.engine.logs import MetricWriter

    writer = MetricWriter(tmp_path)
    for stage, step in (("stage1", 1), ("stage1", 2), ("stage1", 3), ("stage2", 5)):
        writer.write(stage, step, {"x": 1.0})
    assert writer.drop_after("stage1", 2) == 1
    writer.write("stage1", 3, {"x": 2.0})
    writer.close()
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [(r["stage"], r["step"]) for r in rows] == [("stage1", 1), ("stage1", 2), ("stage2", 5), ("stage1", 3)]


class _Indices(torch.utils.data.Dataset):
    """Ten samples that are their own index (module level, so worker processes can unpickle it under any start method)."""

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return index


def _as_list(items):
    return list(items)


def test_batch_stream_resumes_where_it_stopped():
    from cora.engine.streams import BatchStream

    Indices, collate = _Indices, _as_list
    for workers in (0, 2):
        whole = BatchStream(Indices(), 2, collate, workers, seed=3, distributed=False)
        first = [next(whole) for _ in range(9)]
        assert whole.consumed == 9
        whole.close(); whole.close()
        with pytest.raises(RuntimeError, match="closed"):
            next(whole)
        resumed = BatchStream(Indices(), 2, collate, workers, seed=3, distributed=False, skip_batches=4)
        assert resumed.consumed == 4 and [next(resumed) for _ in range(5)] == first[4:]
        resumed.close()
    with pytest.raises(ValueError):
        BatchStream(Indices(), 2, collate, 0, seed=0, distributed=False, skip_batches=-1)


def _gloo_worker(rank, folder):
    """Two CPU processes: disjoint data shares and no_sync gradient accumulation (spawned by the test)."""
    import torch.distributed as dist
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

    from cora.engine.streams import BatchStream
    from cora.engine.trainer import Trainer

    dist.init_process_group("gloo", init_method=f"file://{folder}/rendezvous", rank=rank, world_size=2)

    class Indices(torch.utils.data.Dataset):
        def __len__(self):
            return 12

        def __getitem__(self, index):
            return index % 12

    stream = BatchStream(Indices(), 2, list, 0, seed=7, distributed=True)
    seen = [i for _ in range(3) for i in next(stream)]

    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    net = torch.nn.parallel.DistributedDataParallel(model)
    reduces = []

    def hook(state, bucket):
        reduces.append(1)
        return default_hooks.allreduce_hook(None, bucket)

    net.register_comm_hook(None, hook)
    xs = [torch.full((2, 3), float(rank + 1) * (k + 1)) for k in range(2)]  # a labeled and an unlabeled pass
    trainer = type("T", (), {"distributed": True, "_sync_context": Trainer._sync_context})()
    for k, x in enumerate(xs):
        with trainer._sync_context(net, last_backward=k == 1):
            net(x).sum().backward()
    # reference: each rank's own two-pass gradient, averaged over ranks
    own = torch.nn.Linear(3, 1)
    own.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    for x in xs:
        own(x).sum().backward()
    expected = [p.grad.clone() for p in own.parameters()]
    for p in expected:
        dist.all_reduce(p)
        p /= 2
    ok = all(torch.allclose(a.grad, b, atol=1e-6) for a, b in zip(model.parameters(), expected))
    torch.save({"seen": seen, "reduces": len(reduces), "ok": ok}, f"{folder}/rank{rank}.pt")
    dist.destroy_process_group()


def test_two_processes_read_disjoint_shares_and_reduce_gradients_once(tmp_path):
    torch.multiprocessing.spawn(_gloo_worker, args=(str(tmp_path),), nprocs=2, join=True)
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in range(2))
    assert sorted(r0["seen"] + r1["seen"]) == list(range(12)), "ranks share one permutation, no overlap"
    assert r0["ok"] and r1["ok"], "no_sync accumulation equals the rank-mean of the summed passes"
    assert r0["reduces"] == 1 and r1["reduces"] == 1, "one all-reduce per optimizer step, not one per backward"


class _TwoPassNet(torch.nn.Module):
    """Stand-in for the Stage 3 step: some tensors take part in the labeled pass only (lm_head, the anchor
    encoders), some in the unlabeled pass only, the rest in both."""

    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(3, 4)
        self.only_first = torch.nn.Linear(4, 1)
        self.only_second = torch.nn.Linear(4, 1)

    def forward(self, x, second):
        return (self.only_second if second else self.only_first)(torch.tanh(self.shared(x)))


def _gloo_unused_worker(rank, folder):
    """Gradients of tensors that the last (synchronised) pass does not use must still be averaged over the ranks."""
    import torch.distributed as dist

    from cora.engine.trainer import Trainer

    dist.init_process_group("gloo", init_method=f"file://{folder}/rendezvous", rank=rank, world_size=2)
    torch.manual_seed(0)
    model = _TwoPassNet()
    net = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    trainer = type("T", (), {"distributed": True, "_sync_context": Trainer._sync_context})()
    own = _TwoPassNet()
    ok = []
    for step in range(3):  # several optimizer steps: the bookkeeping of used parameters must reset every step
        xs = [torch.full((2, 3), 0.1 * (rank + 1) * (k + 1) * (step + 1)) for k in range(2)]
        model.zero_grad(set_to_none=True)
        for k, x in enumerate(xs):
            with trainer._sync_context(net, last_backward=k == 1):
                net(x, second=k == 1).sum().backward()
        own.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
        own.zero_grad(set_to_none=True)
        for k, x in enumerate(xs):
            own(x, second=k == 1).sum().backward()
        expected = {n: p.grad.clone() for n, p in own.named_parameters()}
        for value in expected.values():
            dist.all_reduce(value)
            value /= 2
        got = dict(model.named_parameters())
        ok.append({n: bool(torch.allclose(got[n].grad, expected[n], atol=1e-6)) for n in expected})
        with torch.no_grad():  # move the weights so that the next step differs
            for p in model.parameters():
                p -= 0.1 * p.grad
    differs = bool((expected["only_first.weight"] - own.only_first.weight.grad).abs().max() > 1e-4)
    torch.save({"ok": ok, "rank_gradients_differ": differs}, f"{folder}/unused_rank{rank}.pt")
    dist.destroy_process_group()


def test_two_processes_average_gradients_of_tensors_unused_in_the_last_pass(tmp_path):
    """Stage 3 with several GPUs: lm_head and the anchor encoders get gradients from the labeled pass only, which
    runs under ``no_sync``; the all-reduce of the unlabeled pass must still average them over the ranks."""
    torch.multiprocessing.spawn(_gloo_unused_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(tmp_path / f"unused_rank{rank}.pt")
        assert result["rank_gradients_differ"], "the test needs rank-specific gradients to mean anything"
        for step, flags in enumerate(result["ok"]):
            assert all(flags.values()), f"rank {rank}, step {step}: not the rank mean for {[n for n, f in flags.items() if not f]}"


def test_stop_request_sets_a_flag_and_a_second_signal_interrupts():
    import os
    import signal

    from cora.engine.trainer import _StopRequest

    before = signal.getsignal(signal.SIGTERM)
    with _StopRequest() as stop:
        assert stop.signal_name is None
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.signal_name == "SIGTERM"
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
    assert signal.getsignal(signal.SIGTERM) == before


def test_fingerprints_ignore_bookkeeping_and_name_the_changed_key():
    from cora.config import Config
    from cora.engine.preflight import diff_fingerprints, resolve_only_stages, stage_fingerprint

    cfg = Config()
    s1, s3 = cfg.train.stages[0], cfg.train.stages[2]
    base = stage_fingerprint(cfg, s1)
    cfg.train.run_name, cfg.train.log_every, cfg.train.save_every, cfg.data.num_workers = "other", 1, 1, 0
    cfg.unlabeled.weight = 0.1  # decides stage 3 only
    assert stage_fingerprint(cfg, s1) == base and diff_fingerprints(base, stage_fingerprint(cfg, s1)) == []
    cfg.train.lr = 1e-3
    assert diff_fingerprints(base, stage_fingerprint(cfg, s1)) == ["train.lr: 2e-05 -> 0.001"]
    assert "unlabeled" in stage_fingerprint(cfg, s3) and "unlabeled" not in base
    assert stage_fingerprint(cfg, s1, world_size=4)["world_size"] == 4
    # settings that cannot reach a stage are not part of its fingerprint: the teacher of the pseudo-labels and the
    # source of the conditional set do not change stage 1, so runs that differ in them can share it
    cfg = Config()
    s1, s2, s3 = cfg.train.stages
    before = [stage_fingerprint(cfg, s) for s in (s1, s2, s3)]
    cfg.data.ssl_method, cfg.data.conditional_sources, cfg.data.eval_ignore_void = "semivl", "llm", True
    after = [stage_fingerprint(cfg, s) for s in (s1, s2, s3)]
    assert after[0] == before[0]
    assert diff_fingerprints(before[1], after[1]) == ["data.conditional_sources: 'both' -> 'llm'"]
    assert diff_fingerprints(before[2], after[2]) == ["data.conditional_sources: 'both' -> 'llm'",
                                                      "data.ssl_method: 'unimatch' -> 'semivl'"]
    with pytest.raises(ValueError, match="unknown stage"):
        resolve_only_stages(cfg, ["stage9"])


def test_evaluation_helpers(tmp_path):
    from cora.config import Config, save_config
    from cora.engine import evaluation as EV

    # ReasonSeg: a missing long/short flag is an error, a table is named after the paper's columns
    flags = {0: True, 1: None}
    dataset = type("D", (), {"__len__": lambda self: 2, "is_long_query": lambda self, i: flags[i]})()
    with pytest.raises(ValueError, match="paths.reasonseg_manifest"):
        EV.require_query_lengths(dataset, allow_unknown=False)
    assert EV.require_query_lengths(dataset, allow_unknown=True) == 1
    results = {"all": {"ciou": 41.0}, "long": {"ciou": 50.0}, "reasonseg": {"ciou": 41.0}}
    assert EV.table_columns("test", results) == {"Test": {"ciou": 41.0}, "Test (LQ)": {"ciou": 50.0}}
    assert list(EV.table_columns("val", {"all": {"ciou": 1.0}})) == ["Val"]

    # result files of different weights, sets and subsets do not collide
    names = {EV.result_filename("cityscapes_val", "teacher_forced", tag, sets, limit)
             for tag, sets, limit in (("stage3", ["semantic"], None), ("base", ["semantic"], None),
                                      ("stage3", ["attribute", "semantic"], None), ("stage3", ["semantic"], 5))}
    assert len(names) == 4 and "cityscapes_val_teacher_forced__stage3__semantic.json" in names

    # sets named by the user must exist; sets from the configuration are skipped when missing
    cfg = Config()
    cfg.paths.data_dir = str(tmp_path / "data")
    folder = cfg.data_dir / "instructions" / "val"
    folder.mkdir(parents=True)
    (folder / "semantic.jsonl").write_text("")
    assert EV.select_sets(cfg, "val", None) == ["semantic"]
    assert EV.select_sets(cfg, "val", ["semantic"]) == ["semantic"]
    with pytest.raises(FileNotFoundError, match="attribute"):
        EV.select_sets(cfg, "val", ["semantic", "attribute"])

    # --run-dir: the run's own configuration and location
    run = tmp_path / "outputs" / "myrun"
    run.mkdir(parents=True)
    cfg.model.lora_r = 4
    save_config(cfg, str(run / "config.yaml"))
    loaded = EV.load_run_config(None, str(run), ["eval.batch_size=3"])
    assert loaded.model.lora_r == 4 and loaded.eval.batch_size == 3 and loaded.run_dir == run.resolve()
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        EV.load_run_config(None, str(tmp_path / "outputs"), [])

    # tensors the loaded weights do not cover are reported by group
    model = _Tiny()
    assert EV.warn_missing_tensors(model, {"a"}, ["conditional"]) == ["b"]
    assert EV.warn_missing_tensors(model, {"a", "b"}, []) == []

    # a weights file from another model configuration is reported as such
    with pytest.raises(ckpt.CheckpointMismatch, match="config.yaml of the run"):
        ckpt.load_state_checked(_Tiny(), {"zzz": torch.zeros(1)}, "w.pt")


def test_fingerprints_written_before_a_setting_existed_read_it_at_its_neutral_value():
    """Stages trained before data.reasonseg_train_share and the LISA keys existed can still be continued and linked
    under the defaults; a run that uses the new settings differs from them."""
    from cora.engine.preflight import diff_fingerprints

    saved = {"data": {"partition": "1_16"}, "model": {"llava": "llava-hf/llava-1.5-7b-hf"}}
    current = {"data": {"partition": "1_16", "reasonseg_train_share": 0.0},
               "model": {"llava": "llava-hf/llava-1.5-7b-hf", "text_hidden_fcs_init": None, "pretrained_seg_token": False}}
    assert diff_fingerprints(saved, current) == []
    current["data"]["reasonseg_train_share"] = 0.0625
    current["model"]["pretrained_seg_token"] = True
    assert diff_fingerprints(saved, current) == ["data.reasonseg_train_share: 0.0 -> 0.0625", "model.pretrained_seg_token: False -> True"]
    # model.train_seg_projection and data.reasonseg_train_images came later still: absent means the projection was
    # trained and all ReasonSeg train images were used
    current["model"]["train_seg_projection"] = True
    current["data"]["reasonseg_train_images"] = None
    assert diff_fingerprints(saved, current) == ["data.reasonseg_train_share: 0.0 -> 0.0625", "model.pretrained_seg_token: False -> True"]
    current["model"]["train_seg_projection"] = False
    assert "model.train_seg_projection: True -> False" in diff_fingerprints(saved, current)
    current["model"]["train_seg_projection"] = True
    current["data"]["reasonseg_train_images"] = 15
    assert "data.reasonseg_train_images: None -> 15" in diff_fingerprints(saved, current)
    current["data"]["reasonseg_train_images"] = None
    current["data"]["something_else"] = 1  # a key without a recorded neutral value still counts as a difference
    assert "data.something_else: '<absent>' -> 1" in diff_fingerprints(saved, current)
