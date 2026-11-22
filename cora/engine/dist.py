"""Multi-GPU helpers: one process per GPU, started with ``torchrun`` (``python -m torch.distributed.run``)."""
from __future__ import annotations

import os
import random
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.distributed as dist


BACKENDS = ("nccl", "gloo")


def init_distributed() -> Tuple[bool, int, int, torch.device]:
    """Join the process group when launched by torchrun.

    Returns ``(distributed, rank, world_size, device)``. A plain ``python`` launch is a single process
    on the first GPU, or on the CPU when no GPU is visible. Several processes use one GPU each (NCCL);
    ``CORA_DIST_BACKEND=gloo`` runs them on the CPU instead, which the tests use to exercise the multi-process
    code without GPUs.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and "RANK" in os.environ:
        backend = os.environ.get("CORA_DIST_BACKEND", "nccl")
        if backend not in BACKENDS:
            raise ValueError(f"CORA_DIST_BACKEND must be one of {BACKENDS}, got {backend!r}")
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("distributed training needs CUDA devices (CORA_DIST_BACKEND=gloo runs CPU processes)")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return True, dist.get_rank(), dist.get_world_size(), device
    return False, 0, 1, torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def _collective_device() -> str:
    """Device of the small tensors of collective calls: the GPU with NCCL, the CPU with gloo."""
    return "cuda" if dist.get_backend() == "nccl" else "cpu"


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def gather_objects(obj: Any) -> List[Any]:
    """Collect one picklable object from every rank (list ordered by rank)."""
    if not is_distributed():
        return [obj]
    out: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def mean_across_ranks(values: dict) -> dict:
    """Average a dict of floats over ranks (for logging)."""
    if not is_distributed():
        return dict(values)
    keys = sorted(values)
    tensor = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=_collective_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return {k: tensor[i].item() for i, k in enumerate(keys)}


def any_rank(flag: bool) -> bool:
    """True when ``flag`` is true on at least one rank (a collective call: every rank must make it)."""
    if not is_distributed():
        return bool(flag)
    tensor = torch.tensor([1 if flag else 0], dtype=torch.int32, device=_collective_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def capture_rng_state(device: torch.device) -> dict:
    """Python, NumPy, torch and (for a CUDA device) CUDA generator states as tensors and plain values.

    The result loads with ``torch.load(weights_only=True)``, so it can be stored in a checkpoint.
    """
    version, keys, gauss = random.getstate()
    _, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    state = {
        "python": {"version": int(version), "keys": list(keys), "gauss": gauss},
        "numpy": {"keys": torch.from_numpy(np_keys.astype(np.int64)), "pos": int(np_pos),
                  "has_gauss": int(np_has_gauss), "cached_gaussian": float(np_cached)},
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict, device: torch.device) -> None:
    """Inverse of :func:`capture_rng_state`."""
    py = state["python"]
    random.setstate((py["version"], tuple(py["keys"]), py["gauss"]))
    np_state = state["numpy"]
    np.random.set_state(("MT19937", np_state["keys"].numpy().astype(np.uint32), np_state["pos"],
                         np_state["has_gauss"], np_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
