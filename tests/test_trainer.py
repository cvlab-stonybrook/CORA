"""End-to-end test of the three-stage trainer on CPU with the tiny model and a miniature Cityscapes folder."""
import json

import numpy as np
import pytest
import torch
import yaml

from cora import constants as C
from cora.data.adapters import get_adapter
from cora.data.schema import write_jsonl
from cora.engine import checkpoint as ckpt
from cora.engine.trainer import Trainer
from cora.instructions.offline import build_offline_records
from cora.instructions.semantic import build_semantic_records
from cora.instructions.templated_conditional import build_templated_conditional
from cora.pseudo.io import prediction_paths, write_prediction
from tests.fixtures import tiny_config

pytestmark = pytest.mark.data

N_LABELED, N_UNLABELED, N_VAL = 3, 2, 2


def make_config(tmp_path, cityscapes_root, **train):
    """Tiny model, three stages of two steps each, data folder built from a few real Cityscapes frames."""
    cfg = tiny_config()
    cfg.paths.data_dir = str(tmp_path / "data")
    cfg.paths.output_dir = str(tmp_path / "out")
    cfg.paths.cityscapes_root = str(cityscapes_root)
    cfg.data.partition = "1_30"
    cfg.data.ssl_method = "debug"
    cfg.data.num_workers = 0
    cfg.data.queries_per_image = 2
    cfg.unlabeled.num_queries = 3
    cfg.unlabeled.batch_size = 1
    cfg.unlabeled.allow_partial = True
    cfg.train.batch_size = 1
    cfg.train.iters_per_stage = 2
    cfg.train.warmup_steps = 1
    cfg.train.log_every = 1
    cfg.train.save_every = 1
    cfg.train.run_name = "tiny"
    cfg.eval.max_samples = 6
    for key, value in train.items():
        setattr(cfg.train, key, value)

    adapter = get_adapter(cfg)
    labeled = adapter.partition("1_30", "labeled")[:N_LABELED]
    unlabeled = adapter.partition("1_30", "unlabeled")[:N_UNLABELED]
    val = adapter.val_list()[:N_VAL]
    for split, ids in (("train", labeled), ("val", val)):
        write_jsonl(cfg.instructions_path(split, C.SET_SEMANTIC), build_semantic_records(cfg, adapter, split, ids))
        write_jsonl(cfg.instructions_path(split, C.SET_ATTRIBUTE),
                    build_offline_records(cfg, adapter, split, ids, [C.SET_ATTRIBUTE]))
        write_jsonl(cfg.instructions_path(split, C.SET_CONDITIONAL, "templated"),
                    build_templated_conditional(cfg, adapter, split, ids))
    rng = np.random.default_rng(0)
    for image_id in unlabeled:
        label = adapter.load_label(image_id)
        write_prediction(cfg.pseudo_dir, image_id, label, rng.uniform(0.3, 1.0, label.shape).astype(np.float32))
    return cfg


def read_metrics(cfg):
    with open(cfg.run_dir / "metrics.jsonl") as f:
        return [json.loads(line) for line in f]


def test_three_stages_end_to_end(tmp_path, cityscapes_root):
    cfg = make_config(tmp_path, cityscapes_root)
    results = Trainer(cfg).fit()
    assert list(results) == ["stage1", "stage2", "stage3"]
    for name in results:
        stage_dir = cfg.run_dir / name
        assert (stage_dir / "final.pt").is_file() and not (stage_dir / "last.pt").exists()
        metrics = json.loads((stage_dir / "metrics.json").read_text())
        assert metrics["all"]["n"] > 0 and 0.0 <= metrics["all"]["ciou"] <= 100.0
        # the rule-based conditional records are reported under their own name, apart from the LLM-generated set
        assert {"semantic", "attribute", "conditional_templated"} <= set(metrics) and "conditional" not in metrics
    assert (cfg.run_dir / "config.yaml").is_file()

    rows = read_metrics(cfg)
    steps = {name: [r for r in rows if r["stage"] == name and "labeled/loss" in r] for name in results}
    assert all(len(v) == 2 for v in steps.values())
    for row in steps["stage1"] + steps["stage2"]:
        assert "labeled/loss" in row and not any(k.startswith("unlabeled/") for k in row)
    for row in steps["stage3"]:
        # Eq. 9: the unlabeled term is logged scaled by sigma, next to its unscaled parts
        assert row["unlabeled/loss"] == pytest.approx(cfg.unlabeled.weight * (row["unlabeled/seg"] + row["unlabeled/tfca"]), rel=1e-4)
        assert 0.0 <= row["unlabeled/weight_mean"] <= 1.0 and "unlabeled_to_labeled" in row
    assert all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v, float))
    # the logged learning rate is the one the step applied (it used to show the next step's rate: 0 at the end)
    assert all(row["lr"] == pytest.approx(cfg.train.lr) for name in steps for row in steps[name])
    # validation rows carry the step at which they were computed, under the stage's own name
    val_rows = [r for r in rows if "val/all/ciou" in r]
    assert [(r["stage"], r["step"]) for r in val_rows] == [(name, 2) for name in results]
    assert all({"val/all/giou", "val/all/n"} <= set(r) for r in val_rows)

    # Stage 1 trains no anchor encoder; stages with references do.
    names1 = set(ckpt.load_checkpoint(cfg.run_dir / "stage1" / "final.pt")["model"])
    names2 = set(ckpt.load_checkpoint(cfg.run_dir / "stage2" / "final.pt")["model"])
    assert not any(n.startswith("anchors.") for n in names1) and any(n.startswith("anchors.") for n in names2)

    # A second run finds every stage finished and trains nothing.
    assert Trainer(cfg).fit() == {}


def write_reasonseg_train(cfg, count=3):
    """A prepared ReasonSeg train folder (as tools/prepare_reasonseg.py writes it) with a few random images."""
    import cv2

    folder = cfg.reasonseg_train_dir
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    index = []
    for i in range(count):
        image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[10:30, 12 + i : 40] = 1
        cv2.imwrite(str(folder / f"rs{i}.jpg"), image)
        cv2.imwrite(str(folder / f"rs{i}.png"), mask)
        index.append({"image_id": f"rs{i}", "text": f"the thing number {i} that people use every day" if i else "kettle", "is_sentence": bool(i)})
    (folder / "index.json").write_text(json.dumps(index))


def test_reasonseg_train_share_mixes_reasonseg_images_into_every_stage(tmp_path, cityscapes_root):
    from cora.data.datasets import LabeledDataset
    from cora.model.build import build_tokenizer

    cfg = make_config(tmp_path, cityscapes_root)
    _, spec = build_tokenizer(cfg)
    stage = cfg.train.stages[0]
    plain = [LabeledDataset(cfg, stage, spec, seed=0)[i].image_id for i in range(30)]
    assert not any(i.startswith("reasonseg/") for i in plain)
    cfg.data.reasonseg_train_share = 0.5
    with pytest.raises(FileNotFoundError, match="prepare_reasonseg.py --split train"):
        LabeledDataset(cfg, stage, spec, seed=0)
    with pytest.raises(FileNotFoundError, match="prepare_reasonseg.py") as raised:
        Trainer(cfg)  # the preflight finds it before the model is built
    assert "[stage1] labeled data" in str(raised.value) and "needs the instruction sets" not in str(raised.value)
    write_reasonseg_train(cfg)
    mixed = LabeledDataset(cfg, stage, spec, seed=0)
    samples = [mixed[i] for i in range(120)]
    share = sum(s.image_id.startswith("reasonseg/") for s in samples) / len(samples)
    assert 0.35 < share < 0.65 and len(mixed) == len(LabeledDataset(make_config(tmp_path / "x", cityscapes_root), stage, spec))
    assert {c.set_name for s in samples for c in s.conversations} == {"reasonseg", "semantic", "attribute"}
    assert [mixed[i].image_id for i in range(20)] == [s.image_id for s in samples[:20]], "draws depend on the index only"
    # three stages train on the mix (labeled loss, token bank with class -1, stage evaluation)
    results = Trainer(cfg).fit()
    assert list(results) == ["stage1", "stage2", "stage3"]
    rows = [r for r in read_metrics(cfg) if "labeled/loss" in r]
    assert len(rows) == 6 and all(np.isfinite(r["labeled/loss"]) for r in rows)
    # the share is part of what decides a stage's weights: the finished run is not reused under another share
    cfg.data.reasonseg_train_share = 0.25
    with pytest.raises(ValueError, match="data.reasonseg_train_share"):
        Trainer(cfg)


def test_stage_order_is_enforced_and_weights_carry_over(tmp_path, cityscapes_root):
    cfg = make_config(tmp_path, cityscapes_root)
    with pytest.raises(RuntimeError, match="stage1"):
        Trainer(cfg, only_stages=["stage2"]).fit()
    with pytest.raises(ValueError, match="unknown stage"):
        Trainer(cfg, only_stages=["stage9"])
    first = Trainer(cfg, only_stages=["stage1"])
    assert list(first.fit()) == ["stage1"]
    after_stage1 = ckpt.load_checkpoint(cfg.run_dir / "stage1" / "final.pt")["model"]
    second = Trainer(cfg, only_stages=["stage2"])
    seen = {}
    original = second.run_stage

    def spy(stage):
        seen.update({n: p.detach().clone() for n, p in second.model.named_parameters() if n in after_stage1})
        return original(stage)

    second.run_stage = spy
    assert list(second.fit()) == ["stage2"]
    assert seen and all(torch.equal(seen[n].cpu(), after_stage1[n]) for n in seen)


def test_resume_from_last_checkpoint(tmp_path, cityscapes_root, monkeypatch):
    cfg = make_config(tmp_path, cityscapes_root, iters_per_stage=3)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(ckpt, "save_final", interrupted)
    with pytest.raises(KeyboardInterrupt):
        Trainer(cfg, only_stages=["stage1"]).fit()
    last = cfg.run_dir / "stage1" / "last.pt"
    assert ckpt.load_checkpoint(last)["meta"]["step"] == 2
    monkeypatch.undo()

    Trainer(cfg, only_stages=["stage1"]).fit()
    assert (cfg.run_dir / "stage1" / "final.pt").is_file() and not last.exists()
    stage1_steps = [r["step"] for r in read_metrics(cfg) if r["stage"] == "stage1" and "labeled/loss" in r]
    assert stage1_steps == [1, 2, 3]  # the row of step 3 written before the interruption is replaced, not duplicated
    assert "resumed from" in (cfg.run_dir / "train.log").read_text()


def test_confidence_threshold_baseline_and_no_tfca(tmp_path, cityscapes_root):
    """The ablation baseline: thresholded pseudo-labels, no K-1 branch, no token alignment (Table 4 row 1)."""
    cfg = make_config(tmp_path, cityscapes_root)
    cfg.unlabeled.weighting = "confidence"
    cfg.tfca.enabled = False
    cfg.train.stages = [cfg.train.stages[2]]
    cfg.train.stages[0].labeled_sets = [C.SET_SEMANTIC, C.SET_ATTRIBUTE]
    cfg.train.stages[0].use_reference = False
    results = Trainer(cfg).fit()
    rows = [r for r in read_metrics(cfg) if r["stage"] == "stage3" and "labeled/loss" in r]
    assert list(results) == ["stage3"] and len(rows) == 2 and all(r["unlabeled/tfca"] == 0.0 and r["unlabeled/pairs"] == 0.0 for r in rows)


# ----------------------------------------------------------------------------------------------
# Resume equivalence, interruption, safety checks, preflight, evaluation tool
# ----------------------------------------------------------------------------------------------
def final_weights(cfg, stage):
    return ckpt.load_checkpoint(cfg.run_dir / stage / "final.pt")["model"]


def stage3_only(cfg):
    """One unlabeled stage without anchor encoders: labeled stream, unlabeled stream, token bank and teacher."""
    cfg.train.stages = [cfg.train.stages[2]]
    cfg.train.stages[0].labeled_sets = [C.SET_SEMANTIC, C.SET_ATTRIBUTE]
    cfg.train.stages[0].use_reference = False
    return cfg


def interrupt_after_save(monkeypatch, step):
    """Make the run stop right after the ``last.pt`` of ``step`` is written (as a kill would)."""
    original = ckpt.save_last

    def save_and_die(path, model, optimizer, scheduler, bank, teacher, meta, rng=None):
        original(path, model, optimizer, scheduler, bank, teacher, meta, rng)
        if meta["step"] == step:
            raise KeyboardInterrupt

    monkeypatch.setattr(ckpt, "save_last", save_and_die)


@pytest.mark.parametrize("unlabeled", [False, True], ids=["labeled_only", "with_unlabeled_teacher_bank"])
def test_resumed_run_equals_an_uninterrupted_one(tmp_path, cityscapes_root, monkeypatch, unlabeled):
    """4 steps = 2 steps, stop, resume, 2 steps: same batches, same RNG draws, same weights and gradient norms."""
    def config(name):
        cfg = make_config(tmp_path, cityscapes_root, iters_per_stage=4, save_every=2, run_name=name)
        cfg.model.lora_dropout = 0.0
        if unlabeled:
            stage3_only(cfg)
        else:
            cfg.train.stages = cfg.train.stages[:1]
        cfg.eval.max_samples = 1
        return cfg

    straight = config("straight")
    Trainer(straight).fit()

    resumed = config("resumed")
    with monkeypatch.context() as patch:
        interrupt_after_save(patch, 2)
        with pytest.raises(KeyboardInterrupt):
            Trainer(resumed).fit()
    stage = resumed.train.stages[0].name
    saved = ckpt.load_checkpoint(ckpt.find_last(resumed.run_dir, stage))
    assert saved["meta"]["step"] == 2 and saved["meta"]["labeled_batches"] == 2 and len(saved["rng"]) == 1
    Trainer(resumed).fit()

    a, b = final_weights(straight, stage), final_weights(resumed, stage)
    assert set(a) == set(b)
    assert all(torch.allclose(a[n], b[n], atol=1e-7, rtol=0) for n in a), "weights differ after resuming"
    norms = {name: [r["grad_norm"] for r in read_metrics(cfg) if r["stage"] == stage and "grad_norm" in r]
             for name, cfg in (("straight", straight), ("resumed", resumed))}
    assert norms["straight"] == pytest.approx(norms["resumed"], rel=1e-5) and len(norms["resumed"]) == 4


def fail_torch_save(monkeypatch, failing_steps):
    """Make ``torch.save`` fail like a full disk (a partial file, then RuntimeError) for the ``last.pt`` of some steps."""
    original = torch.save
    attempts = []

    def save(obj, path, *args, **kwargs):
        step = obj.get("meta", {}).get("step") if "optimizer" in obj else None
        if step is not None:
            attempts.append(step)
            if step in failing_steps and attempts.count(step) == 1:
                with open(path, "wb") as f:
                    f.write(b"partial")
                raise RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed")
        return original(obj, path, *args, **kwargs)

    monkeypatch.setattr(ckpt.torch, "save", save)
    return attempts


def test_a_failed_periodic_save_does_not_stop_the_run(tmp_path, cityscapes_root, monkeypatch):
    def config(name):
        cfg = make_config(tmp_path, cityscapes_root, iters_per_stage=4, save_every=1, run_name=name)
        cfg.model.lora_dropout = 0.0
        cfg.train.stages = cfg.train.stages[:1]
        cfg.eval.max_samples = 1
        return cfg

    straight = config("straight")
    Trainer(straight).fit()
    disturbed = config("disturbed")
    with monkeypatch.context() as patch:
        attempts = fail_torch_save(patch, {1, 2})
        Trainer(disturbed).fit()
    assert attempts == [1, 2, 3]  # every periodic save was tried once; the run went on after the two failures
    stage_dir = disturbed.run_dir / "stage1"
    assert (stage_dir / "final.pt").is_file() and not list(stage_dir.glob("*.tmp")) and not (stage_dir / "last.pt").exists()
    text = (disturbed.run_dir / "train.log").read_text()
    assert "could not write last.pt at step 1" in text and "there is no earlier one" in text
    assert "could not write last.pt at step 2" in text and "could not write last.pt at step 3" not in text
    a, b = final_weights(straight, "stage1"), final_weights(disturbed, "stage1")
    assert all(torch.equal(a[n], b[n]) for n in a), "a failed save changed the training"

    # the previous last.pt survives a failed save and still resumes the run
    kept = config("kept")
    kept.train.iters_per_stage = 3  # saves at steps 1 and 2 (the second fails), then the run is cut before final.pt

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        fail_torch_save(patch, {2})
        patch.setattr(ckpt, "save_final", interrupted)
        with pytest.raises(KeyboardInterrupt):
            Trainer(kept).fit()
    last = ckpt.find_last(kept.run_dir, "stage1")
    assert ckpt.load_checkpoint(last)["meta"]["step"] == 1 and "the previous one stays" in (kept.run_dir / "train.log").read_text()


def test_a_stop_signal_in_the_step_of_a_failed_save_writes_the_state_again(tmp_path, cityscapes_root, monkeypatch):
    import os
    import signal

    from cora.engine.logs import MetricWriter
    from cora.engine.trainer import TrainingInterrupted

    cfg = make_config(tmp_path, cityscapes_root, iters_per_stage=3, save_every=1)
    cfg.train.stages = cfg.train.stages[:1]
    original = MetricWriter.write
    sent = []

    def write_and_signal(self, stage, step, scalars, global_step=None):
        original(self, stage, step, scalars, global_step)
        if step == 1 and "labeled/loss" in scalars and not sent:
            sent.append(step)
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(MetricWriter, "write", write_and_signal)
    attempts = fail_torch_save(monkeypatch, {1})
    with pytest.raises(TrainingInterrupted, match="Run the same command again"):
        Trainer(cfg).fit()
    monkeypatch.undo()
    assert attempts == [1, 1]  # the periodic save failed, the save required by the stop succeeded
    assert ckpt.load_checkpoint(ckpt.find_last(cfg.run_dir, "stage1"))["meta"]["step"] == 1
    assert list(Trainer(cfg).fit()) == ["stage1"]


def test_a_stage_started_again_from_its_first_step_replaces_its_metric_rows(tmp_path, cityscapes_root, monkeypatch):
    """final.pt cannot be written and there is no last.pt (train.save_every=0): the rerun trains the stage again, and
    metrics.jsonl must hold every step of the stage once, not the rows of both attempts."""
    cfg = make_config(tmp_path, cityscapes_root, iters_per_stage=3, save_every=0)
    cfg.train.stages = cfg.train.stages[:1]

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(ckpt, "save_final", full_disk)
        with pytest.raises(OSError, match="No space left"):
            Trainer(cfg).fit()
    first = [r["step"] for r in read_metrics(cfg) if r["stage"] == "stage1" and "labeled/loss" in r]
    assert first == [1, 2, 3] and ckpt.find_last(cfg.run_dir, "stage1") is None
    assert list(Trainer(cfg).fit()) == ["stage1"]
    again = [r["step"] for r in read_metrics(cfg) if r["stage"] == "stage1" and "labeled/loss" in r]
    assert again == [1, 2, 3] and "removed 3 metric rows of an earlier attempt" in (cfg.run_dir / "train.log").read_text()


def test_sigterm_saves_the_state_and_the_run_resumes(tmp_path, cityscapes_root, monkeypatch):
    import os
    import signal

    from cora.engine.logs import MetricWriter
    from cora.engine.trainer import TrainingInterrupted

    cfg = make_config(tmp_path, cityscapes_root, iters_per_stage=3, save_every=0)
    cfg.train.stages = cfg.train.stages[:1]
    original = MetricWriter.write
    sent = []

    def write_and_signal(self, stage, step, scalars, global_step=None):
        original(self, stage, step, scalars, global_step)
        if step == 1 and "labeled/loss" in scalars and not sent:
            sent.append(step)
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(MetricWriter, "write", write_and_signal)
    with pytest.raises(TrainingInterrupted, match="Run the same command again") as raised:
        Trainer(cfg).fit()
    assert raised.value.exit_code == 143
    monkeypatch.undo()
    last = ckpt.find_last(cfg.run_dir, "stage1")
    assert last is not None and ckpt.load_checkpoint(last)["meta"]["step"] == 1  # save_every=0: written by the interrupt
    assert list(Trainer(cfg).fit()) == ["stage1"] and ckpt.find_final(cfg.run_dir, "stage1") is not None


def test_non_finite_gradients_skip_the_step_and_repeated_ones_stop_the_run(tmp_path, cityscapes_root, monkeypatch):
    cfg = make_config(tmp_path, cityscapes_root, iters_per_stage=4)
    cfg.train.stages = cfg.train.stages[:1]
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = []

    def clip_nan_once(params, clip):
        norm = real_clip(params, clip)
        calls.append(1)
        return torch.tensor(float("nan")) if len(calls) == 2 else norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_nan_once)
    Trainer(cfg).fit()
    log = (cfg.run_dir / "train.log").read_text()
    assert "step 2: non-finite gradient norm, optimizer step skipped" in log
    assert all(np.isfinite(r["grad_norm"]) for r in read_metrics(cfg) if "grad_norm" in r)
    monkeypatch.undo()

    broken = make_config(tmp_path, cityscapes_root, iters_per_stage=8, run_name="broken", lr=1e30, grad_clip=0.0)
    broken.train.stages = broken.train.stages[:1]
    with pytest.raises(FloatingPointError, match="NaN|non-finite"):
        Trainer(broken).fit()
    assert ckpt.find_final(broken.run_dir, "stage1") is None, "NaN weights must never become final.pt"


def test_a_stage_without_metrics_is_evaluated_on_the_next_run(tmp_path, cityscapes_root):
    cfg = make_config(tmp_path, cityscapes_root)
    cfg.train.stages = cfg.train.stages[:1]
    Trainer(cfg).fit()
    metrics = cfg.run_dir / "stage1" / "metrics.json"
    metrics.unlink()  # the run died between final.pt and the evaluation
    results = Trainer(cfg).fit()
    assert list(results) == ["stage1"] and metrics.is_file() and results["stage1"]["all"]["n"] > 0
    assert Trainer(cfg).fit() == {}


def test_finished_stages_are_not_reused_under_other_settings(tmp_path, cityscapes_root):
    cfg = make_config(tmp_path, cityscapes_root)
    cfg.train.stages = cfg.train.stages[:1]
    Trainer(cfg).fit()
    changed = make_config(tmp_path, cityscapes_root)
    changed.train.stages = changed.train.stages[:1]
    changed.train.lr = 1e-3
    with pytest.raises(ValueError, match=r"train\.lr: 2e-05 -> 0\.001") as raised:
        Trainer(changed)
    assert "train.run_name" in str(raised.value) and "--allow-config-change" in str(raised.value)
    assert yaml.safe_load((cfg.run_dir / "config.yaml").read_text())["train"]["lr"] == 2e-05, "the first configuration stays"
    assert Trainer(changed, allow_config_change=True).fit() == {}
    assert len(list(cfg.run_dir.glob("config.*.yaml"))) == 1, "the differing invocation is recorded next to it"

    again = make_config(tmp_path, cityscapes_root, resume=False)
    again.train.stages = again.train.stages[:1]
    with pytest.raises(ValueError, match="train.resume=false"):
        Trainer(again)


def test_resume_with_a_changed_learning_rate_is_refused(tmp_path, cityscapes_root, monkeypatch):
    cfg = make_config(tmp_path, cityscapes_root, iters_per_stage=3, save_every=1)
    cfg.train.stages = cfg.train.stages[:1]
    with monkeypatch.context() as patch:
        interrupt_after_save(patch, 1)
        with pytest.raises(KeyboardInterrupt):
            Trainer(cfg).fit()
    changed = make_config(tmp_path, cityscapes_root, iters_per_stage=3, save_every=1, lr=5e-4)
    changed.train.stages = changed.train.stages[:1]
    with pytest.raises(ValueError, match="last.pt"):
        Trainer(changed)
    shorter = make_config(tmp_path, cityscapes_root, iters_per_stage=1)
    shorter.train.stages = shorter.train.stages[:1]
    with pytest.raises(ValueError, match="steps"):
        Trainer(shorter, allow_config_change=True).fit()


def test_problems_surface_before_the_model_is_built_or_any_stage_trained(tmp_path, cityscapes_root, monkeypatch):
    from cora.engine import trainer as trainer_module

    def no_model(*args, **kwargs):
        raise AssertionError("the model must not be built when the preflight fails")

    monkeypatch.setattr(trainer_module, "build_model", no_model)
    cfg = make_config(tmp_path, cityscapes_root)
    # stage order: every missing stage is named at once and nothing is trained
    with pytest.raises(RuntimeError, match=r"\['stage1', 'stage2'\].*--stages stage1,stage2,stage3"):
        Trainer(cfg, only_stages=["stage3"])
    # missing instruction file of stage 2
    cfg.instructions_path("train", C.SET_CONDITIONAL, "templated").unlink()
    with pytest.raises(FileNotFoundError, match=r"\[stage2\] labeled data"):
        Trainer(cfg)
    assert not list(cfg.run_dir.glob("stage*/final.pt"))
    # missing pseudo-labels of the unlabeled images: the message names the keys to set
    cfg = make_config(tmp_path, cityscapes_root, run_name="partial")
    first = get_adapter(cfg).partition("1_30", "unlabeled")[0]
    prediction_paths(cfg.pseudo_dir, first)[0].unlink()
    cfg.unlabeled.allow_partial = False
    with pytest.raises(FileNotFoundError, match=r"data\.ssl_method.*unlabeled\.allow_partial=true"):
        Trainer(cfg, only_stages=["stage1", "stage2", "stage3"])


def test_streams_are_closed_before_the_stage_evaluation(tmp_path, cityscapes_root, monkeypatch):
    import multiprocessing

    from cora.engine import trainer as trainer_module

    cfg = stage3_only(make_config(tmp_path, cityscapes_root))
    cfg.data.num_workers = 2
    seen = {}
    original = trainer_module.Trainer.evaluate

    def spy(self, stage, step, final):
        seen[(stage.name, final)] = len(multiprocessing.active_children())
        return original(self, stage, step, final)

    monkeypatch.setattr(trainer_module.Trainer, "evaluate", spy)
    Trainer(cfg).fit()
    assert seen == {("stage3", True): 0}, "the training loaders' workers must be gone when the evaluation starts"


def test_evaluation_leaves_the_random_generators_untouched(tmp_path, cityscapes_root):
    """A data loader draws its seed from the global generator; an evaluation between two training steps must not
    change what the next steps draw (a run resumed from the last.pt of that step would otherwise differ)."""
    import random

    from cora.engine import dist as D

    cfg = make_config(tmp_path, cityscapes_root)
    trainer = Trainer(cfg)
    before = D.capture_rng_state(trainer.device)
    draws = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    D.restore_rng_state(before, trainer.device)
    results = trainer.evaluate(cfg.train.stages[0], 1, final=False)
    assert results["all"]["n"] == 6
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == draws


def test_stage_evaluation_uses_its_own_sample_limit(tmp_path, cityscapes_root):
    cfg = make_config(tmp_path, cityscapes_root)
    cfg.train.stages = cfg.train.stages[:1]
    cfg.eval.max_samples, cfg.eval.stage_max_samples = None, 3
    results = Trainer(cfg).fit()
    assert results["stage1"]["all"]["n"] == 3  # eval.stage_max_samples limits the evaluations during training
    again = make_config(tmp_path, cityscapes_root, run_name="fallback")
    again.train.stages = again.train.stages[:1]
    again.eval.max_samples, again.eval.stage_max_samples = 4, None
    assert Trainer(again).fit()["stage1"]["all"]["n"] == 4  # without it, eval.max_samples applies


def run_two_processes(tmp_path, arguments, timeout=900):
    """Start ``python <arguments>`` twice on CPU with the environment that torchrun sets for two processes (gloo
    backend); returns the two exit codes (None for a process that had to be killed after ``timeout`` seconds)."""
    import os
    import socket
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    processes, logs = [], []
    for rank in range(2):
        env = dict(os.environ, CORA_DIST_BACKEND="gloo", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(repo), RANK=str(rank),
                   LOCAL_RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        logs.append(open(tmp_path / f"rank{rank}.log", "w"))
        processes.append(subprocess.Popen([sys.executable, *[str(a) for a in arguments]],
                                          env=env, stdout=logs[-1], stderr=subprocess.STDOUT))
    codes = []
    try:
        for p in processes:
            try:
                codes.append(p.wait(timeout=timeout))
            except subprocess.TimeoutExpired:
                codes.append(None)
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
        for handle in logs:
            handle.close()
    return codes


def rank_logs(tmp_path):
    return "\n".join(f"--- rank {r}\n" + (tmp_path / f"rank{r}.log").read_text()[-3000:] for r in range(2))


def test_two_cpu_processes_train_all_stages_and_hold_the_same_weights(tmp_path, cityscapes_root):
    """The whole trainer with two CPU processes (gloo), started with the environment that torchrun sets: gradient
    synchronisation over the two backward passes of a stage 3 step, the gathered token bank, the collective
    checkpoint, the sharded evaluation, and the end-of-stage check that both processes hold identical weights."""
    from pathlib import Path

    from cora.config import save_config

    cfg = make_config(tmp_path, cityscapes_root, run_name="ddp")
    config_file = tmp_path / "ddp.yaml"
    save_config(cfg, str(config_file))
    repo = Path(__file__).resolve().parents[1]
    codes = run_two_processes(tmp_path, [repo / "tools" / "train.py", "--config", config_file])
    assert codes == [0, 0], rank_logs(tmp_path)
    log = (cfg.run_dir / "train.log").read_text()
    assert log.count("the 2 processes hold identical weights") == 3, log[-3000:]
    for stage in ("stage1", "stage2", "stage3"):
        assert (cfg.run_dir / stage / "final.pt").is_file() and not (cfg.run_dir / stage / "last.pt").exists()
        # both processes evaluate their share of the 6 samples and the statistics are summed
        assert json.loads((cfg.run_dir / stage / "metrics.json").read_text())["all"]["n"] == 6
    rows = [r for r in read_metrics(cfg) if r["stage"] == "stage3" and "labeled/loss" in r]
    assert len(rows) == 2 and all(np.isfinite(r["unlabeled/loss"]) and np.isfinite(r["grad_norm"]) for r in rows)
    fingerprint = json.loads((cfg.run_dir / "stage3" / "fingerprint.json").read_text())
    assert fingerprint["world_size"] == 2


FAILING_SAVE = '''
"""tools/train.py with a main process that cannot write its checkpoints. MODE=stop: last.pt fails and SIGTERM
reaches the other process in step 1 (the save that the stop requires then fails). MODE=type: the periodic save fails
with a TypeError. MODE=final: last.pt is written, final.pt fails."""
import os, runpy, signal, sys
import torch
from cora.engine import checkpoint as ckpt
from cora.engine import trainer as T

rank, mode, config, tool = int(os.environ["RANK"]), sys.argv[1], sys.argv[2], sys.argv[3]
original = torch.save

def save(obj, path, *args, **kwargs):
    is_last = isinstance(obj, dict) and "optimizer" in obj
    if rank == 0 and is_last and mode in ("stop", "type"):
        raise TypeError("cannot pickle") if mode == "type" else OSError(28, "No space left on device")
    if rank == 0 and mode == "final" and isinstance(path, (str, os.PathLike)) and "final.pt" in str(path):
        raise OSError(28, "No space left on device")
    return original(obj, path, *args, **kwargs)

ckpt.torch.save = save
inner, sent = T.Trainer._save_last, []

def _save_last(self, stage, stage_dir, done, *args, **kwargs):
    if mode == "stop" and rank == 1 and done == 1 and not sent:
        sent.append(done)
        os.kill(os.getpid(), signal.SIGTERM)
    return inner(self, stage, stage_dir, done, *args, **kwargs)

T.Trainer._save_last = _save_last
sys.argv = ["train.py", "--config", config, "--stages", "stage1"]
runpy.run_path(tool, run_name="__main__")
'''


@pytest.mark.parametrize("mode", ["stop", "type", "final"])
def test_a_fatal_save_failure_ends_both_processes(tmp_path, cityscapes_root, mode):
    """The main process cannot write last.pt when the run is told to stop (or fails with an error that is not
    about writing), or cannot write final.pt: it must not leave alone, or the other process would wait in its next
    collective call."""
    from pathlib import Path

    from cora.config import save_config

    cfg = make_config(tmp_path, cityscapes_root, run_name="fatal", iters_per_stage=3, save_every=1)
    config_file = tmp_path / "fatal.yaml"
    save_config(cfg, str(config_file))
    wrapper = tmp_path / "failing_save.py"
    wrapper.write_text(FAILING_SAVE)
    repo = Path(__file__).resolve().parents[1]
    codes = run_two_processes(tmp_path, [wrapper, mode, config_file, repo / "tools" / "train.py"], timeout=300)
    assert None not in codes, "a process was still waiting after 300 s:\n" + rank_logs(tmp_path)
    assert all(code != 0 for code in codes), rank_logs(tmp_path)
    told = "the main process could not write final.pt" if mode == "final" else "the main process could not write last.pt at step 1"
    assert told in (tmp_path / "rank1.log").read_text(), rank_logs(tmp_path)
    expected = "cannot pickle" if mode == "type" else "No space left on device"
    assert expected in (tmp_path / "rank0.log").read_text(), rank_logs(tmp_path)
    assert not (cfg.run_dir / "stage1" / "final.pt").exists() and not list((cfg.run_dir / "stage1").glob("*.tmp"))
    if mode == "final":  # the state of the last periodic save is kept, so the stage can still be finished
        assert ckpt.load_checkpoint(cfg.run_dir / "stage1" / "last.pt")["meta"]["step"] == 2
        codes = run_two_processes(tmp_path, [repo / "tools" / "train.py", "--config", config_file, "--stages", "stage1"])
        assert codes == [0, 0] and (cfg.run_dir / "stage1" / "final.pt").is_file(), rank_logs(tmp_path)


def test_warnings_for_batch_size_and_a_dominant_unlabeled_term(tmp_path, cityscapes_root, monkeypatch):
    from cora.engine import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "UNLABELED_RATIO_WARNING", 0.0)
    cfg = stage3_only(make_config(tmp_path, cityscapes_root))
    Trainer(cfg).fit()
    log = (cfg.run_dir / "train.log").read_text()
    assert "effective labeled batch size is 1" in log and "the paper uses 8" in log and "train.grad_accum_steps=8" in log
    assert log.count("times the labeled loss") == 1 and "unlabeled.reduction='sum'" in log


def test_ablation_run_links_the_stages_it_shares(tmp_path, cityscapes_root):
    first = make_config(tmp_path, cityscapes_root)
    Trainer(first, only_stages=["stage1", "stage2"]).fit()
    second = make_config(tmp_path, cityscapes_root, run_name="sigma")
    second.unlabeled.weight = 0.1  # decides stage 3 only
    results = Trainer(second, init_stages_from=first.run_dir).fit()
    assert list(results) == ["stage3"]
    for stage in ("stage1", "stage2"):
        link = second.run_dir / stage / "final.pt"
        assert link.is_symlink() and link.resolve() == (first.run_dir / stage / "final.pt").resolve()
        assert (second.run_dir / stage / "metrics.json").is_file()
    # settings that reach a shared stage make it ineligible: stage 1 is trained again
    third = make_config(tmp_path, cityscapes_root, run_name="lr", lr=1e-4)
    assert list(Trainer(third, only_stages=["stage1"], init_stages_from=first.run_dir).fit()) == ["stage1"]
    assert not (third.run_dir / "stage1" / "final.pt").is_symlink()


def test_pruned_run_still_reloads_and_evaluates(tmp_path, cityscapes_root):
    cfg = make_config(tmp_path, cityscapes_root)
    Trainer(cfg, prune_checkpoints=True).fit()
    assert [ckpt.find_final(cfg.run_dir, s) is None for s in ("stage1", "stage2", "stage3")] == [True, True, False]
    assert ckpt.is_pruned(cfg.run_dir, "stage1") and ckpt.is_finished(cfg.run_dir, "stage2")
    assert Trainer(cfg).fit() == {}


def load_tool(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_tool_options_and_failure_logging(tmp_path, cityscapes_root, capsys):
    from cora.config import save_config

    cfg = make_config(tmp_path, cityscapes_root)
    config_file = tmp_path / "cfg.yaml"
    save_config(cfg, str(config_file))
    tool = load_tool("train")
    # overrides may follow the options; --check builds no model and trains nothing
    tool.main(["--config", str(config_file), "--check", "--stages", "stage1", "train.run_name=checked"])
    assert "check passed: finished [], to train ['stage1']" in capsys.readouterr().out
    assert not list((cfg.run_dir.parent / "checked").glob("stage*/final.pt"))
    tool.main(["--config", str(config_file), "train.run_name=cli", "--stages", "stage1", "train.iters_per_stage=1"])
    assert (cfg.run_dir.parent / "cli" / "stage1" / "final.pt").is_file()
    with pytest.raises(RuntimeError, match="stage1"):
        tool.main(["--config", str(config_file), "train.run_name=cli2", "--stages", "stage2"])
    assert "training failed" in (cfg.run_dir.parent / "cli2" / "train.log").read_text()


def test_evaluate_tool_names_its_results_and_refuses_unusable_input(tmp_path, cityscapes_root, monkeypatch, capsys):
    import json

    cfg = make_config(tmp_path, cityscapes_root)
    Trainer(cfg, only_stages=["stage1", "stage2"]).fit()
    tool = load_tool("evaluate")
    run = str(cfg.run_dir)
    tool.main(["--run-dir", run, "--sets", "semantic", "--max-samples", "2"])
    tool.main(["--run-dir", run, "--sets", "semantic", "--max-samples", "2", "--no-checkpoint"])
    tool.main(["--run-dir", run, "--sets", "semantic", "--stage", "stage1", "--max-samples", "2"])
    files = sorted(p.name for p in (cfg.run_dir / "eval").glob("*.json"))
    assert files == ["cityscapes_val_teacher_forced__base__semantic__n2.json",
                     "cityscapes_val_teacher_forced__stage1__semantic__n2.json",
                     "cityscapes_val_teacher_forced__stage2__semantic__n2.json"]
    report = json.loads((cfg.run_dir / "eval" / files[2]).read_text())
    assert report["sets"] == ["semantic"] and report["samples_evaluated"] == 2 and report["max_samples"] == 2
    assert report["samples_total"] > 2 and report["results"]["all"]["n"] == 2 and report["config"]["model"]["tiny"]
    assert "weights:" in capsys.readouterr().out
    # a stage checkpoint of the run itself is filed under its stage; the same stage of ANOTHER run folder gets its own
    # tag, so it does not overwrite this run's result
    import shutil

    other = cfg.run_dir.parent / "other"
    shutil.copytree(cfg.run_dir, other, ignore=shutil.ignore_patterns("eval"))
    own_file = cfg.run_dir / "eval" / files[2]
    before = own_file.read_bytes()
    tool.main(["--run-dir", run, "--sets", "semantic", "--max-samples", "2", "--checkpoint", str(other / "stage2" / "final.pt")])
    foreign = cfg.run_dir / "eval" / "cityscapes_val_teacher_forced__other-stage2__semantic__n2.json"
    assert foreign.is_file() and own_file.read_bytes() == before
    assert json.loads(foreign.read_text())["weights_tag"] == "other-stage2"
    tool.main(["--run-dir", run, "--sets", "semantic", "--max-samples", "2", "--checkpoint", str(cfg.run_dir / "stage2" / "final.pt")])
    assert json.loads(own_file.read_text())["weights_tag"] == "stage2"
    capsys.readouterr()
    # a set from the configuration that does not exist is skipped, one the user names is an error
    (cfg.data_dir / "instructions" / "val" / "attribute.jsonl").unlink()
    tool.main(["--run-dir", run, "--max-samples", "2", "--output", str(tmp_path / "x.json")])
    with pytest.raises(FileNotFoundError, match="attribute"):
        tool.main(["--run-dir", run, "--sets", "attribute", "--max-samples", "1"])
    # option checks
    for bad in (["--checkpoint", "a.pt", "--no-checkpoint"], ["--max-samples", "0"], ["--max-samples", "-1"]):
        with pytest.raises(SystemExit):
            tool.main(["--run-dir", run, *bad])
    # ReasonSeg without the long/short flag: an error before the model is built, not half a table
    from cora.data import reasonseg

    class NoFlags:
        def __init__(self, cfg, spec, split="test"):
            pass

        def __len__(self):
            return 3

        def is_long_query(self, index):
            return None

    monkeypatch.setattr(reasonseg, "ReasonSegDataset", NoFlags)
    with pytest.raises(ValueError, match="paths.reasonseg_manifest"):
        tool.main(["--run-dir", run, "--reasonseg", "test"])
