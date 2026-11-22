"""Shared pytest configuration.

Markers:
  gpu   needs a CUDA device and the real model weights; runs only with CORA_TEST_GPU=1
  data  needs a dataset on disk; such tests use the fixtures below, which skip when the data is absent
"""
import os
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device and the real model weights (CORA_TEST_GPU=1)")
    config.addinivalue_line("markers", "data: needs a dataset on disk")


def pytest_collection_modifyitems(config, items):
    switches = {"gpu": "CORA_TEST_GPU"}
    for item in items:
        for marker, variable in switches.items():
            if marker in item.keywords and os.environ.get(variable) != "1":
                item.add_marker(pytest.mark.skip(reason=f"set {variable}=1 to run"))


@pytest.fixture(scope="session")
def cityscapes_root() -> Path:
    from cora.config import Config

    cfg = Config()
    root = Path(os.environ.get("CORA_CITYSCAPES_ROOT", cfg.resolve(cfg.paths.cityscapes_root)))
    if not (root / "leftImg8bit").is_dir() or not (root / "gtFine").is_dir():
        pytest.skip(f"Cityscapes not found at {root}")
    return root


@pytest.fixture(scope="session")
def reasonseg_root() -> Path:
    from cora.config import Config

    cfg = Config()
    root = Path(os.environ.get("CORA_REASONSEG_ROOT", cfg.resolve(cfg.paths.reasonseg_root)))
    if not root.is_dir():
        pytest.skip(f"ReasonSeg not found at {root}")
    return root
