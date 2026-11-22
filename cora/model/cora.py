"""The CORA network: LISA re-implemented on LLaVA-1.5 + SAM with the ``<SSEG>`` token (paper Sec. 3.1).

Eq. 1: the multimodal LLM reads the image and a (possibly conditional) instruction and writes a response that
contains one ``<SSEG>`` token. Eq. 2: SAM's frozen encoder embeds the image. Eq. 3: SAM's decoder turns the
image embedding and the LLM state of the ``<SSEG>`` token into a binary mask.

Following LISA, the vector fed to the decoder is the last-layer state at the position that PREDICTS ``<SSEG>``
(one before the token itself), projected by ``text_hidden_fcs``. Image features are merged by hand
(``inputs_embeds``), because the anchor embeddings of conditional queries overwrite further placeholders.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cora import constants as C
from cora.config import Config
from cora.data.batch import Batch
from cora.model.llava import clip_cls_features, image_features
from cora.model.ref_encoders import AnchorEncoders, place_anchor_embeddings
from cora.model.sam import decode_masks, encode_images
from cora.tokens import TokenSpec, setup_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class CoraOutput:
    """Result of :meth:`CoraModel.forward`."""

    ce_loss: Optional[torch.Tensor]  # mean LM loss over the supervised tokens of the batch (None if not requested)
    mask_logits: List[torch.Tensor]  # N tensors (H_i, W_i) float32 logits at ORIGINAL image resolution, sequence order
    seg_hidden: torch.Tensor  # (N, hidden) float32: last-layer LLM state at the position that predicts <SSEG>
    seg_embed: torch.Tensor  # (N, out_dim) float32: projection of seg_hidden = the SAM sparse prompt
    sam_embeddings: torch.Tensor  # (B, 256, h, w) float32 frozen SAM image embeddings (reusable)


@contextmanager
def eval_mode(module: nn.Module) -> Iterator[None]:
    """Put ``module`` in eval mode and restore the training flag of every submodule afterwards."""
    flags = [(m, m.training) for m in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for m, flag in flags:
            m.training = flag


class CoraModel(nn.Module):
    """LLaVA-1.5 (+ LoRA) and SAM joined by the ``<SSEG>`` token, plus the optional anchor encoders (Sec. 3.3).

    Instances are assembled by :func:`cora.model.build.build_model`, which also sets the trainable flags.
    """

    def __init__(
        self,
        cfg: Config,
        spec: TokenSpec,
        llava: nn.Module,
        sam: nn.Module,
        text_hidden_fcs: nn.Module,
        anchors: Optional[AnchorEncoders] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.llava = llava
        self.sam = sam
        self.text_hidden_fcs = text_hidden_fcs
        self.anchors = anchors
        self._tokenizer = None

    # ------------------------------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype of the frozen language model (bf16 or fp32)."""
        return self.llava.model.language_model.norm.weight.dtype

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.cfg.model.precision == "bf16"
        )

    def attach_tokenizer(self, tokenizer) -> None:
        """Tokenizer used by :meth:`generate` to decode the generated text."""
        self._tokenizer = tokenizer

    def _get_tokenizer(self):
        if self._tokenizer is None:
            tokenizer, spec = setup_tokenizer(self.cfg.llava_source, self.spec.image_seq_len, self.spec.clip_image_size)
            if spec != self.spec:
                raise RuntimeError("the tokenizer built from the config does not match this model's TokenSpec")
            self._tokenizer = tokenizer
        return self._tokenizer

    def _validate(self, batch: Batch) -> None:
        if batch.input_ids.device != self.device:
            raise ValueError(f"batch is on {batch.input_ids.device} but the model is on {self.device}; call batch.to(device)")
        n, length = batch.input_ids.shape
        for name in ("attention_mask", "labels"):
            if getattr(batch, name).shape != (n, length):
                raise ValueError(f"batch.{name} has shape {tuple(getattr(batch, name).shape)}, expected {(n, length)}")
        b = batch.images_sam.shape[0]
        if batch.images_clip.shape[0] != b or len(batch.resize_hw) != b or len(batch.orig_hw) != b:
            raise ValueError("images_sam, images_clip, resize_hw and orig_hw must all describe the same B images")
        if batch.seq_image.shape != (n,) or (n and (int(batch.seq_image.min()) < 0 or int(batch.seq_image.max()) >= b)):
            raise ValueError("batch.seq_image must hold one valid image index per sequence")

    # ------------------------------------------------------------------------------------------
    # Pieces of the forward pass
    # ------------------------------------------------------------------------------------------
    def _input_embeddings(self, batch: Batch, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings with the image features and the anchor embeddings written over their placeholders."""
        spec = self.spec
        embeddings = self.llava.get_input_embeddings()(input_ids).to(self.dtype)
        image_positions = input_ids == spec.image_token_id
        counts = image_positions.sum(1)
        if bool((counts != spec.image_seq_len).any()):
            raise ValueError(
                f"every sequence needs exactly {spec.image_seq_len} image tokens, found {counts.tolist()}"
            )
        features = image_features(self.llava, batch.images_clip)  # CLIP runs once per image
        embeddings = embeddings.masked_scatter(
            image_positions.unsqueeze(-1), features[batch.seq_image].to(embeddings.dtype)
        )
        num_refs = int(batch.ref_seq.shape[0])
        if num_refs:
            if self.anchors is None:
                raise ValueError(
                    "the batch has reference objects but the model has no anchor encoders "
                    "(model.reference.mode='text' builds none)"
                )
            mask_embeddings, box_embeddings = self.anchors(clip_cls_features(self.llava, batch.ref_pixels), batch.ref_boxes)
        else:
            mask_embeddings = box_embeddings = embeddings.new_zeros(0, embeddings.shape[-1])
        embeddings = place_anchor_embeddings(
            embeddings, input_ids, spec, batch.ref_seq, mask_embeddings, box_embeddings
        )
        if not num_refs and self.anchors is not None:
            zero = self.anchors.zero_term()  # keeps unused trainable anchor parameters in the graph (DDP)
            if zero is not None:
                embeddings = embeddings + zero.to(embeddings.dtype)
        return embeddings

    def _run_llm(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last-layer states ``(N, L, hidden)`` (after the final RMSNorm) of the language model."""
        return self.llava.model.language_model(
            inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state

    def _lm_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over the supervised tokens; the output head is applied to those positions only."""
        targets = labels[:, 1:]
        selected = targets != C.IGNORE_INDEX
        if not bool(selected.any()):
            return hidden.float().sum() * 0.0  # all labels ignored: a zero that keeps the graph
        logits = self.llava.get_output_embeddings()(hidden[:, :-1][selected])
        return F.cross_entropy(logits.float(), targets[selected])

    def _seg_positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Index of the single ``<SSEG>`` token of every sequence (raises when a sequence has none or several)."""
        is_seg = input_ids == self.spec.seg_token_id
        counts = is_seg.sum(1)
        bad = (counts != 1).nonzero(as_tuple=True)[0]
        if bad.numel():
            raise ValueError(
                f"every sequence must contain exactly one {C.SEG_TOKEN}; sequences {bad.tolist()} "
                f"contain {counts[bad].tolist()}"
            )
        positions = is_seg.int().argmax(dim=1)
        if bool((positions < 1).any()) or bool((attention_mask.gather(1, positions[:, None]) == 0).any()):
            raise ValueError(f"{C.SEG_TOKEN} must follow at least one token and lie inside the attention mask")
        return positions

    def _project(self, seg_hidden: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=seg_hidden.device.type, enabled=False):
            return self.text_hidden_fcs(seg_hidden.float())

    def _decode(
        self, seg_embed: torch.Tensor, rows: torch.Tensor, batch: Batch, sam_embeddings: torch.Tensor
    ) -> List[torch.Tensor]:
        """Decode one mask per prompt; ``rows`` are the sequence indices of the prompts. One image embedding
        is reused for all sequences of an image."""
        images = batch.seq_image[rows]
        masks: List[Optional[torch.Tensor]] = [None] * int(rows.numel())
        for image in torch.unique(images).tolist():
            where = (images == image).nonzero(as_tuple=True)[0]
            decoded = decode_masks(
                self.sam, sam_embeddings[image : image + 1], seg_embed[where], batch.resize_hw[image], batch.orig_hw[image]
            )
            for j, index in enumerate(where.tolist()):
                masks[index] = decoded[j]
        return masks  # type: ignore[return-value]

    # ------------------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_sam(self, images_sam: torch.Tensor) -> torch.Tensor:
        """Eq. 2: frozen SAM image embeddings ``(B, 256, h, w)`` float32 for SAM-preprocessed images."""
        with self._autocast():
            return encode_images(self.sam, images_sam)

    def forward(
        self, batch: Batch, compute_text_loss: bool = True, sam_embeddings: Optional[torch.Tensor] = None
    ) -> CoraOutput:
        """Eq. 1-3 over a batch. ``sam_embeddings`` may be passed in to avoid recomputing the frozen encoder."""
        self._validate(batch)
        if sam_embeddings is None:
            sam_embeddings = self.encode_sam(batch.images_sam)
        elif sam_embeddings.shape[0] != batch.images_sam.shape[0]:
            raise ValueError("sam_embeddings must hold one embedding per image of the batch")
        with self._autocast():
            embeddings = self._input_embeddings(batch, batch.input_ids)
            hidden = self._run_llm(embeddings, batch.attention_mask)
            ce_loss = self._lm_loss(hidden, batch.labels) if compute_text_loss else None
        positions = self._seg_positions(batch.input_ids, batch.attention_mask)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        seg_hidden = hidden[rows, positions - 1].float()  # LISA: the state before <SSEG>
        seg_embed = self._project(seg_hidden)
        mask_logits = self._decode(seg_embed, rows, batch, sam_embeddings)
        return CoraOutput(
            ce_loss=ce_loss,
            mask_logits=mask_logits,
            seg_hidden=seg_hidden,
            seg_embed=seg_embed,
            sam_embeddings=sam_embeddings,
        )

    @torch.no_grad()
    def generate(self, batch: Batch, max_new_tokens: int = 32) -> Tuple[List[str], List[Optional[torch.Tensor]]]:
        """Greedy answers for prompts that end in ``ASSISTANT:``, then the mask of each answer's first ``<SSEG>``.

        The masks come from one teacher-forced pass over prompt + generated ids (robust to the KV cache), cut
        right after the first ``<SSEG>``; sequences that produced none get ``None``.
        """
        self._validate(batch)
        if bool((batch.input_ids == self.spec.seg_token_id).any()):
            raise ValueError(f"generate() needs prompts without an answer, but the input contains {C.SEG_TOKEN}")
        tokenizer = self._get_tokenizer()
        spec = self.spec
        with eval_mode(self), self._autocast():
            sam_embeddings = self.encode_sam(batch.images_sam)
            generated = self._greedy(self._input_embeddings(batch, batch.input_ids), batch.attention_mask, max_new_tokens)
            prompt_lengths = batch.attention_mask.sum(1).tolist()
            sequences, seg_index = [], []
            for n, new_ids in enumerate(generated):
                if spec.seg_token_id in new_ids:
                    cut = new_ids.index(spec.seg_token_id)
                    seg_index.append(prompt_lengths[n] + cut)
                    new_ids = new_ids[: cut + 1]
                else:
                    seg_index.append(-1)
                prompt = batch.input_ids[n, : prompt_lengths[n]]
                sequences.append(torch.cat([prompt, torch.tensor(new_ids, dtype=prompt.dtype, device=prompt.device)]))
            texts = [tokenizer.decode(ids, skip_special_tokens=False).strip() for ids in generated]
            masks: List[Optional[torch.Tensor]] = [None] * len(generated)
            rows = torch.tensor([n for n, i in enumerate(seg_index) if i >= 0], dtype=torch.long, device=self.device)
            if rows.numel():
                max_len = max(int(s.shape[0]) for s in sequences)
                ids = torch.full((len(sequences), max_len), spec.pad_token_id, dtype=batch.input_ids.dtype, device=self.device)
                attention = torch.zeros_like(ids)
                for n, seq in enumerate(sequences):
                    ids[n, : seq.shape[0]] = seq
                    attention[n, : seq.shape[0]] = 1
                hidden = self._run_llm(self._input_embeddings(batch, ids), attention)
                positions = torch.tensor([seg_index[n] for n in rows.tolist()], dtype=torch.long, device=self.device)
                seg_embed = self._project(hidden[rows, positions - 1].float())
                for n, mask in zip(rows.tolist(), self._decode(seg_embed, rows, batch, sam_embeddings)):
                    masks[n] = mask
        return texts, masks

    def _greedy(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int) -> List[List[int]]:
        """Greedy decoding from prompt embeddings (right-padded input, left-padded internally, KV cache).

        Returns the generated ids of every sequence, cut before the first EOS.
        """
        from transformers import DynamicCache

        spec = self.spec
        n, _, hidden_size = embeddings.shape
        lengths = attention_mask.sum(1)
        if not torch.equal(attention_mask, (torch.arange(attention_mask.shape[1], device=lengths.device) < lengths[:, None]).long()):
            raise ValueError("generate() expects right-padded prompts (attention_mask of 1s followed by 0s)")
        width = int(lengths.max())
        left = embeddings.new_zeros(n, width, hidden_size)
        mask = torch.zeros(n, width, dtype=attention_mask.dtype, device=embeddings.device)
        for i, length in enumerate(lengths.tolist()):
            left[i, width - length :] = embeddings[i, :length]
            mask[i, width - length :] = 1
        positions = (mask.cumsum(1) - 1).clamp(min=0)
        banned_ids = [spec.image_token_id, spec.ref_mask_token_id, spec.ref_box_token_id, spec.pad_token_id, spec.bos_token_id]
        if spec.lisa_layout:
            banned_ids += [spec.im_start_token_id, spec.im_end_token_id]
        banned = torch.tensor(banned_ids, device=embeddings.device)
        language_model = self.llava.model.language_model
        embed = self.llava.get_input_embeddings()
        head = self.llava.get_output_embeddings()
        cache = DynamicCache()
        out = language_model(
            inputs_embeds=left, attention_mask=mask, position_ids=positions, past_key_values=cache, use_cache=True
        )
        last = positions[:, -1]
        finished = torch.zeros(n, dtype=torch.bool, device=embeddings.device)
        steps: List[torch.Tensor] = []
        for step in range(max_new_tokens):
            logits = head(out.last_hidden_state[:, -1]).float()
            logits[:, banned] = float("-inf")
            logits[:, spec.vocab_size :] = float("-inf")  # unused padding rows of the embedding matrix
            token = logits.argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, spec.pad_token_id), token)
            steps.append(token)
            finished = finished | (token == spec.eos_token_id)
            if bool(finished.all()) or step == max_new_tokens - 1:
                break
            mask = torch.cat([mask, torch.ones(n, 1, dtype=mask.dtype, device=mask.device)], dim=1)
            last = last + 1
            out = language_model(
                inputs_embeds=embed(token[:, None]).to(embeddings.dtype),
                attention_mask=mask,
                position_ids=last[:, None],
                past_key_values=cache,
                use_cache=True,
            )
        result = torch.stack(steps, dim=1).tolist() if steps else [[] for _ in range(n)]
        return [ids[: ids.index(spec.eos_token_id)] if spec.eos_token_id in ids else ids for ids in result]

    # ------------------------------------------------------------------------------------------
    # Trainable state
    # ------------------------------------------------------------------------------------------
    def set_reference_trainable(self, flag: bool) -> None:
        """The anchor encoders train only in stages that use references (Sec. 3.3)."""
        if self.anchors is None:
            if flag:
                logger.warning("set_reference_trainable(True): the model has no anchor encoders (reference.mode='text')")
            return
        self.anchors.set_trainable(flag)

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Every parameter with ``requires_grad`` as a CPU tensor (LoRA, embeddings, head, projection, decoder,
        anchor encoders). Frozen weights are not included; the dict holds tensors only."""
        return {n: p.detach().to("cpu", copy=True) for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
        """Copy a :meth:`trainable_state_dict` into the model.

        ``strict=True`` requires exactly the currently trainable names (so a stage-1 checkpoint cannot be
        loaded silently into a model whose anchor encoders are trainable). With ``strict=False`` every name
        of the model is loaded, even a frozen one; missing and unknown names are logged.
        """
        params = dict(self.named_parameters())
        trainable = {n for n, p in params.items() if p.requires_grad}
        missing = sorted(trainable - set(state))
        unexpected = sorted(set(state) - (trainable if strict else set(params)))
        if strict and (missing or unexpected):
            raise KeyError(f"trainable state mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        if not strict:
            if missing:
                logger.warning("%d trainable tensors are absent from the state and keep their values", len(missing))
            if unexpected:
                logger.warning("ignoring %d state entries that are not parameters of this model", len(unexpected))
        with torch.no_grad():
            for name, value in state.items():
                if name not in params:
                    continue
                if params[name].shape != value.shape:
                    raise ValueError(f"{name}: shape {tuple(value.shape)} in the state, {tuple(params[name].shape)} in the model")
                params[name].copy_(value)
