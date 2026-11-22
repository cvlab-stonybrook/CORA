"""The network that handles the K-1 rephrasings of an unlabeled class query (paper Sec. 3.2.1, Fig. 3).

The K-1 queries pass through a LLM and decoder that are not trained by them; their predictions only define the
per-pixel weights (Eq. 6). What "frozen" means is not fixed by the paper, so three modes
are offered:

* ``online``: the current model in eval mode without gradients (default);
* ``frozen``: a snapshot of every trainable tensor taken when the unlabeled stage starts;
* ``ema``: an exponential moving average of the trained tensors, updated after each optimizer step.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, Iterator, List

import torch

from cora.data.batch import Batch
from cora.model.cora import CoraModel, eval_mode

logger = logging.getLogger(__name__)

MODES = ("online", "frozen", "ema")


class Teacher:
    """Wrapper that produces mask logits for the K-1 branch from a :class:`CoraModel`.

    The snapshot is swapped into the live parameters only for the duration of a forward pass (storage swap,
    restored in a ``finally`` block), so the live weights and the optimizer are never modified.
    """

    def __init__(self, model: CoraModel, mode: str, ema_momentum: float = 0.999) -> None:
        if mode not in MODES:
            raise ValueError(f"teacher mode must be one of {MODES}, got {mode!r}")
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1), got {ema_momentum}")
        self.model = model
        self.mode = mode
        self.ema_momentum = float(ema_momentum)
        self._snapshot: Dict[str, torch.Tensor] = {}
        self._started = False

    # ------------------------------------------------------------------------------------------
    def _trainable(self) -> Dict[str, torch.Tensor]:
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def start(self) -> None:
        """Call once when the unlabeled stage begins: takes the snapshot of the trainable tensors (frozen, ema)."""
        if self.mode != "online":
            self._snapshot = {n: p.detach().clone() for n, p in self._trainable().items()}
        self._started = True

    @contextmanager
    def _snapshot_swapped_in(self) -> Iterator[None]:
        params = dict(self.model.named_parameters())
        live: Dict[str, torch.Tensor] = {}
        try:
            for name, snap in self._snapshot.items():
                param = params[name]
                live[name] = param.data
                param.data = snap
            yield
        finally:
            for name, data in live.items():
                params[name].data = data

    @torch.no_grad()
    def predict(self, batch: Batch, sam_embeddings: torch.Tensor) -> List[torch.Tensor]:
        """Mask logits (one ``(H, W)`` tensor per sequence of ``batch``) without gradients, in eval mode.

        The training flags of the model are restored afterwards; no loss is computed.
        """
        if not self._started:
            raise RuntimeError("Teacher.start() must be called before predict()")
        with eval_mode(self.model):
            if self.mode == "online":
                out = self.model(batch, compute_text_loss=False, sam_embeddings=sam_embeddings)
            else:
                with self._snapshot_swapped_in():
                    out = self.model(batch, compute_text_loss=False, sam_embeddings=sam_embeddings)
        return [m.detach() for m in out.mask_logits]

    @torch.no_grad()
    def update(self) -> None:
        """After every optimizer step: move the EMA snapshot toward the live weights (no-op for other modes)."""
        if self.mode != "ema":
            return
        if not self._started:
            raise RuntimeError("Teacher.start() must be called before update()")
        momentum = self.ema_momentum
        for name, param in self._trainable().items():
            if name in self._snapshot:
                self._snapshot[name].mul_(momentum).add_(param.detach(), alpha=1.0 - momentum)
            else:  # became trainable after start() (anchor encoders): begin from the current value
                self._snapshot[name] = param.detach().clone()

    # ------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Tensors and primitives only (loadable with ``weights_only=True``)."""
        return {
            "mode": self.mode,
            "ema_momentum": self.ema_momentum,
            "started": self._started,
            "snapshot": {n: t.detach().to("cpu", copy=True) for n, t in self._snapshot.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("mode") != self.mode:
            raise ValueError(f"checkpoint holds a {state.get('mode')!r} teacher, this one is {self.mode!r}")
        params = dict(self.model.named_parameters())
        snapshot: Dict[str, torch.Tensor] = {}
        for name, tensor in state["snapshot"].items():
            if name not in params:
                raise KeyError(f"teacher snapshot has an unknown parameter {name!r}")
            if params[name].shape != tensor.shape:
                raise ValueError(f"{name}: snapshot shape {tuple(tensor.shape)} != model shape {tuple(params[name].shape)}")
            snapshot[name] = tensor.to(device=params[name].device, dtype=params[name].dtype, copy=True)
        self._snapshot = snapshot
        self._started = bool(state.get("started", True))
