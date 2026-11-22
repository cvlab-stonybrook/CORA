"""Construction of the tokenizer and of the CORA network (paper Sec. 3.1, Fig. 3, Sec. 4.1)."""
from __future__ import annotations

import logging
from typing import Tuple

import torch
from torch import nn

from cora.config import Config
from cora.model.cora import CoraModel
from cora.model.llava import (
    add_lora,
    ensure_embedding_rows,
    image_geometry,
    init_new_token_rows,
    load_llava,
    make_embeddings_trainable,
)
from cora.model.ref_encoders import AnchorEncoders
from cora.model.sam import build_sam, configure_sam_trainable
from cora.tokens import TokenSpec, setup_tokenizer

logger = logging.getLogger(__name__)


def build_tokenizer(cfg: Config) -> Tuple[object, TokenSpec]:
    """LLaVA tokenizer with ``<SSEG>``, ``<REFMASK>`` and ``<REFBOX>``; geometry from the LLaVA config.

    The tiny model reuses the real tokenizer files (small, from the HF cache) with the tiny image geometry.
    """
    image_seq_len, clip_image_size = image_geometry(cfg)
    return setup_tokenizer(cfg.llava_source, image_seq_len=image_seq_len, clip_image_size=clip_image_size)


def build_model(cfg: Config, spec: TokenSpec) -> CoraModel:
    """Assemble LLaVA + LoRA + SAM + ``text_hidden_fcs`` (+ anchor encoders) with the trainable flags of Fig. 3.

    Trainable: LoRA on the language model, input embeddings and output head (fp32 copies, for the new
    tokens), ``text_hidden_fcs``, SAM's mask decoder (without its IoU head) and the anchor encoders.
    Frozen: CLIP, LLaVA's projector and base weights, SAM's image and prompt encoders. The model is returned
    on the CPU in training mode; the new token rows are the mean of the pretrained rows. With a converted LISA
    checkpoint (``model.llava`` = its folder, ``model.pretrained_seg_token``, ``model.text_hidden_fcs_init``,
    ``paths.sam_checkpoint``; tools/convert_lisa.py) the ``<SSEG>`` rows of the embedding and of the output head
    stay LISA's trained ``[SEG]`` rows and ``text_hidden_fcs`` starts from LISA's, whatever ``train_embeddings`` is;
    only the two anchor tokens get the mean row.

    Deviation from the drawing of Fig. 3: the figure shows the MLLM frozen apart from LoRA, but the paper builds
    on LISA (Sec. 4.1), which also trains ``embed_tokens`` and ``lm_head`` in full so that the new ``<SSEG>`` row
    (Sec. 3.1) can be learned, as in LISA. Consequences: the trainable state (and so the
    checkpoint, ``CoraModel.trainable_state_dict``) holds the complete embedding and head matrices
    (about 2 x 32064 x hidden values for LLaVA-1.5), not only the new rows. ``model.train_embeddings=false``
    restores the figure literally (LoRA only), at the price that ``<SSEG>`` keeps its mean-of-vocabulary row
    (with a converted LISA checkpoint it keeps LISA's trained row). The figure does not draw ``text_hidden_fcs``
    either; ``model.train_seg_projection=false`` freezes it at its starting values (``model.text_hidden_fcs_init``),
    so that with both switches off exactly what the figure marks trains: LoRA and the mask decoder (and, in the
    stages with references, the anchor encoders of Sec. 3.3, which are new and have no pretrained values).
    """
    llava = load_llava(cfg)
    expected = image_geometry(cfg)
    if (spec.image_seq_len, spec.clip_image_size) != expected:
        raise ValueError(f"TokenSpec geometry {(spec.image_seq_len, spec.clip_image_size)} does not match LLaVA's {expected}")
    if int(llava.config.image_token_id) != spec.image_token_id:
        raise ValueError(f"LLaVA expects image token id {llava.config.image_token_id}, the tokenizer uses {spec.image_token_id}")

    llava.requires_grad_(False)
    ensure_embedding_rows(llava, spec.vocab_size)
    num_lora = add_lora(llava, cfg)  # PEFT freezes everything but the adapters, so it runs before the next step
    if cfg.model.train_embeddings:
        make_embeddings_trainable(llava)  # fp32 before the new rows are set, so they are exact fp32 means
    init_new_token_rows(llava, spec, keep_seg=cfg.model.pretrained_seg_token)
    if cfg.model.gradient_checkpointing:
        llava.gradient_checkpointing_enable()

    sam = build_sam(cfg)
    configure_sam_trainable(sam, cfg.model.train_mask_decoder)

    hidden = llava.config.text_config.hidden_size
    # LISA's text_hidden_fcs: Linear(h, h), ReLU, Linear(h, out_dim), Dropout(0.0).
    text_hidden_fcs = nn.Sequential(
        nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, cfg.model.out_dim), nn.Dropout(0.0)
    )
    if cfg.model.text_hidden_fcs_init:
        path = cfg.resolve(cfg.model.text_hidden_fcs_init)
        if not path.is_file():
            raise FileNotFoundError(f"model.text_hidden_fcs_init: {path} does not exist (tools/convert_lisa.py writes it)")
        text_hidden_fcs.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        logger.info("the <SSEG> projection starts from %s", path)
    if not cfg.model.train_seg_projection:
        text_hidden_fcs.requires_grad_(False)  # Fig. 3 read literally: only LoRA and the mask decoder train
    anchors = None
    if cfg.model.reference.mode == "tokens":
        ref = cfg.model.reference
        anchors = AnchorEncoders(
            hidden, llava.model.multi_modal_projector, ref.mask_projector, ref.modality_dropout, ref.train_projector
        )
    model = CoraModel(cfg, spec, llava, sam, text_hidden_fcs, anchors)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "built CORA model: %d LoRA modules, %.1f M trainable of %.1f M parameters",
        num_lora,
        trainable / 1e6,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model
