"""Endless, distributed-aware batch stream for training (paper Sec. 3.3: fixed number of optimizer steps per stage)."""
from __future__ import annotations

from typing import Callable, Iterator, List

import cv2
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from cora.data.batch import Batch


_WRAP_BASE = 1 << 32  # first index block used for wrapped positions


class InfiniteSampler(Sampler):
    """Yields shuffled passes over ``range(length)`` forever, one pass per epoch.

    The index of epoch ``e`` is ``e * length + permutation[i]``, so datasets that use the index as a random seed
    (:mod:`cora.data.datasets`) never see the same index twice; ``index % length`` is the position in the
    dataset. With ``world_size > 1`` each pass is cut into per-rank shares that are disjoint within the pass
    (the pass is wrapped to a multiple of ``world_size``, so every rank gets the same count).
    """

    def __init__(self, length: int, seed: int, rank: int = 0, world_size: int = 1) -> None:
        if length < 1:
            raise ValueError("cannot sample from an empty dataset")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        self.length, self.seed, self.rank, self.world_size = length, seed, rank, world_size

    def __iter__(self) -> Iterator[int]:
        epoch = 0
        while True:
            generator = torch.Generator()
            generator.manual_seed(self.seed * 1_000_003 + epoch)
            order: List[int] = torch.randperm(self.length, generator=generator).tolist()
            indices = [epoch * self.length + position for position in order]
            # Positions repeated to fill the last round get an index of their own (same position modulo
            # ``length``), so that two ranks never draw with the same seed.
            extra = 0
            while len(indices) % self.world_size:
                position = order[extra % self.length]
                indices.append((_WRAP_BASE + epoch * self.world_size + extra) * self.length + position)
                extra += 1
            yield from indices[self.rank :: self.world_size]
            epoch += 1


def _worker_init(_worker_id: int) -> None:
    cv2.setNumThreads(0)  # one OpenCV thread per worker process


def infinite_loader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Callable,
    num_workers: int,
    seed: int,
    distributed: bool,
) -> Iterator[Batch]:
    """Batches of ``dataset`` forever, reshuffled every pass.

    With ``distributed=True`` the current ``torch.distributed`` rank reads its own share of every pass, so
    ranks draw different samples. Workers are persistent, so there is no start-up cost between passes. The
    dataset must accept any non-negative index (the labeled and unlabeled datasets do).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    rank, world = 0, 1
    if distributed:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("distributed=True needs an initialised torch.distributed process group")
        rank, world = dist.get_rank(), dist.get_world_size()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=InfiniteSampler(len(dataset), seed, rank, world),
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        drop_last=True,
    )
    yield from loader
