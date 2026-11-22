"""Tests of the shipped configuration files: every YAML under configs/ loads, and the four partition files agree."""
from pathlib import Path

import pytest

from cora.config import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"
ALL_CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))
PARTITIONS = ("1_30", "1_16", "1_8", "1_4")


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_every_config_loads(path):
    cfg = load_config(str(path))
    assert cfg.data.dataset == "cityscapes"


@pytest.mark.parametrize("partition", PARTITIONS)
def test_partition_files_resolve_to_the_paper_setting(partition):
    cfg = load_config(str(CONFIG_DIR / "cityscapes" / f"{partition}.yaml"))
    assert cfg.data.dataset == "cityscapes"
    assert cfg.data.partition == partition
    assert cfg.data.ssl_method == "semivl"
    assert cfg.data.reasonseg_train_share == 0.25
    assert cfg.data.reasonseg_train_images is None
    assert cfg.data.clip_preprocess == "center_crop"
    assert cfg.model.train_embeddings is False
    assert cfg.model.train_seg_projection is False
    assert cfg.model.pretrained_seg_token is True
    assert cfg.model.llava == "pretrained/lisa-7b-v1/llava"
    assert cfg.unlabeled.reduction == "sum"
    assert cfg.unlabeled.weight == 0.001
    assert len(cfg.train.stages) == 3
    assert cfg.train.iters_per_stage == 5000
    assert all(stage.iters in (None, 5000) for stage in cfg.train.stages)


def test_partition_files_have_distinct_run_names():
    names = [load_config(str(CONFIG_DIR / "cityscapes" / f"{p}.yaml")).train.run_name for p in PARTITIONS]
    assert len(set(names)) == len(PARTITIONS)
