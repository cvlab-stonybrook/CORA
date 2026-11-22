"""Resumable batch stream for the training loops (paper Sec. 3.3: a fixed number of optimizer steps per stage).

:func:`cora.data.samplers.infinite_loader` always starts at the first batch of the first pass. A stage that is
resumed from ``last.pt`` must continue the stream where the interrupted run stopped, so :class:`BatchStream`
skips the indices of the batches that were already consumed. Skipping happens on the sampler, in the main
process, so no image is read for the skipped batches.
"""
from __future__ import annotations

import itertools
from typing import Callable, Iterator, Optional

import cv2
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from cora.data.batch import Batch
from cora.data.samplers import InfiniteSampler


class _SkipSampler(Sampler):
    """Another sampler's index stream without its first ``skip`` indices."""

    def __init__(self, base: Sampler, skip: int) -> None:
        self.base, self.skip = base, skip

    def __iter__(self) -> Iterator[int]:
        return itertools.islice(iter(self.base), self.skip, None)


def _worker_init(_worker_id: int) -> None:
    cv2.setNumThreads(0)  # one OpenCV thread per worker process


class BatchStream:
    """Endless batches of ``dataset`` that can start after ``skip_batches`` already consumed batches.

    The batches are those of :func:`cora.data.samplers.infinite_loader` with the same arguments (same sampler,
    same ``drop_last`` batching). Every rank must use the same ``seed``: the shuffled pass is shared and cut
    into disjoint per-rank shares by :class:`InfiniteSampler`. The underlying iterator, and so the worker
    processes, is created in the constructor: the draw of the loader's base seed from the global torch RNG then
    happens at a fixed point of the stage set-up, before an RNG state is restored.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        collate_fn: Callable,
        num_workers: int,
        seed: int,
        distributed: bool,
        skip_batches: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        if skip_batches < 0:
            raise ValueError(f"skip_batches must be non-negative, got {skip_batches}")
        rank, world = 0, 1
        if distributed:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError("distributed=True needs an initialised torch.distributed process group")
            rank, world = dist.get_rank(), dist.get_world_size()
        sampler: Sampler = InfiniteSampler(len(dataset), seed, rank, world)
        if skip_batches:
            sampler = _SkipSampler(sampler, skip_batches * batch_size)
        self._loader: Optional[DataLoader] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            worker_init_fn=_worker_init if num_workers > 0 else None,
            drop_last=True,
        )
        self._iterator: Optional[Iterator[Batch]] = iter(self._loader)
        self.consumed = skip_batches  # batches of the stream handed out so far (including the skipped ones)

    def __iter__(self) -> "BatchStream":
        return self

    def __next__(self) -> Batch:
        if self._iterator is None:
            raise RuntimeError("the batch stream is closed")
        batch = next(self._iterator)
        self.consumed += 1
        return batch

    def close(self) -> None:
        """Stop the worker processes and drop the prefetched batches (idempotent)."""
        iterator, self._iterator, self._loader = self._iterator, None, None
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
