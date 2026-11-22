"""Run logging: console + ``train.log``, a JSON-lines metrics file and TensorBoard scalars (main process only)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional


def setup_logger(run_dir: Path, main_process: bool, name: str = "cora") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    if not main_process:
        logger.setLevel(logging.WARNING)
        logger.addHandler(logging.StreamHandler(sys.stderr))
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(Path(run_dir) / "train.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def _beyond(line: str, stage: str, step: int) -> bool:
    row = json.loads(line)
    return row.get("stage") == stage and int(row.get("step", 0)) > step


class MetricWriter:
    """Appends scalars to ``<run_dir>/metrics.jsonl`` and to TensorBoard when it is installed."""

    def __init__(self, run_dir: Path, enabled: bool = True):
        self.enabled = enabled
        self._file = None
        self._tb = None
        if not enabled:
            return
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._path = run_dir / "metrics.jsonl"
        self._file = open(self._path, "a", buffering=1)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        except ImportError:
            self._tb = None

    def write(self, stage: str, step: int, scalars: Dict[str, float], global_step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        self._file.write(json.dumps({"stage": stage, "step": step, **scalars}) + "\n")
        if self._tb is not None:
            for key, value in scalars.items():
                self._tb.add_scalar(f"{stage}/{key}", value, step if global_step is None else global_step)

    def drop_after(self, stage: str, step: int) -> int:
        """Remove the rows of ``stage`` that lie beyond ``step`` from ``metrics.jsonl``; returns how many.

        A stage resumed from a checkpoint at ``step`` writes those rows again; without this they would appear
        twice. TensorBoard event files are append-only, so there the later points are overwritten in the plot.
        """
        if not self.enabled:
            return 0
        self._file.close()
        with open(self._path) as f:
            rows = [line for line in f if line.strip()]
        kept = [line for line in rows if not _beyond(line, stage, step)]
        with open(self._path, "w") as f:
            f.writelines(kept)
        self._file = open(self._path, "a", buffering=1)
        return len(rows) - len(kept)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        if self._tb is not None:
            self._tb.close()
