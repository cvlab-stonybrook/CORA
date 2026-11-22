"""LLaVA-1.5 backbone of CORA: loading, new-token rows, LoRA, trainable flags and feature helpers.

CORA builds LISA on LLaVA-v1.5-7B (paper Sec. 4.1): CLIP ViT-L/14-336 and the MLP projector stay frozen,
the language model gets LoRA (Fig. 3), and the input embeddings and the output head stay trainable so that
the new ``<SSEG>`` token (Sec. 3.1, inherited from LISA) can be learned. Module paths follow
transformers 5.x (``llava.model.vision_tower`` / ``multi_modal_projector`` / ``language_model``).

The same code runs on a converted LISA checkpoint (tools/convert_lisa.py; ``model.llava`` names its folder): there
the backbone is LLaVA-Lightning-7B-v1-1 with CLIP ViT-L/14 at 224 px (256 image tokens) and one linear projector,
stored as ``linear_1`` followed by an identity activation and an identity ``linear_2``.
"""
from __future__ import annotations

import logging
import re
from typing import Tuple

import torch

from cora.config import Config
from cora.tokens import TokenSpec

logger = logging.getLogger(__name__)

# Geometry of the tiny random LLaVA used by unit tests and CPU smoke runs (model.tiny=True).
TINY_CLIP_IMAGE_SIZE = 28
TINY_PATCH_SIZE = 14
TINY_HIDDEN_SIZE = 32
TINY_NUM_LAYERS = 2
# The real checkpoint has 32064 embedding rows for a 32002-token vocabulary; the tiny model mirrors that so
# that the added tokens (ids 32002 ...) fit without resizing.
TINY_EMBEDDING_ROWS = 32064
IMAGE_TOKEN_ID_LLAVA = 32000


def compute_dtype(cfg: Config) -> torch.dtype:
    """Dtype of the frozen LLaVA weights: bf16 or fp32 (``model.precision``)."""
    return torch.bfloat16 if cfg.model.precision == "bf16" else torch.float32


def tiny_llava_config():
    """A random, CPU-sized LLaVA configuration (4 image tokens per 28 px image)."""
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig

    vision = CLIPVisionConfig(
        hidden_size=TINY_HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        image_size=TINY_CLIP_IMAGE_SIZE,
        patch_size=TINY_PATCH_SIZE,
        projection_dim=16,
    )
    text = LlamaConfig(
        vocab_size=TINY_EMBEDDING_ROWS,
        hidden_size=TINY_HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=TINY_NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    )
    return LlavaConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=IMAGE_TOKEN_ID_LLAVA,
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        tie_word_embeddings=False,
    )


def image_geometry(cfg: Config) -> Tuple[int, int]:
    """``(image_seq_len, clip_image_size)`` of the LLaVA in ``cfg`` (576 and 336 for LLaVA-1.5-7B)."""
    if cfg.model.tiny:
        vision = tiny_llava_config().vision_config
    else:
        from transformers import AutoConfig

        llava_config = AutoConfig.from_pretrained(cfg.llava_source)
        if llava_config.vision_feature_select_strategy != "default":
            raise ValueError(
                "only vision_feature_select_strategy='default' (patch tokens without CLS) is supported, "
                f"got {llava_config.vision_feature_select_strategy!r}"
            )
        vision = llava_config.vision_config
    side = vision.image_size // vision.patch_size
    return side * side, int(vision.image_size)


def load_llava(cfg: Config):
    """Pretrained LLaVA-1.5 (or a tiny random one) in the precision of ``cfg``, on the CPU."""
    from transformers import LlavaForConditionalGeneration

    dtype = compute_dtype(cfg)
    if cfg.model.tiny:
        return LlavaForConditionalGeneration(tiny_llava_config()).to(dtype)
    return LlavaForConditionalGeneration.from_pretrained(cfg.llava_source, dtype=dtype, low_cpu_mem_usage=True)


def ensure_embedding_rows(llava, vocab_size: int) -> None:
    """Guarantee at least ``vocab_size`` embedding rows without ever shrinking the matrices.

    LLaVA-1.5 ships 32064 rows for 32002 tokens, so the three CORA tokens fit; ``resize_token_embeddings``
    with ``len(tokenizer)`` would cut the matrices down to 32005 rows and is therefore never called then.
    """
    rows = llava.get_input_embeddings().weight.shape[0]
    if vocab_size > rows:
        llava.resize_token_embeddings(vocab_size, pad_to_multiple_of=64)
        logger.info("grew the embedding matrix from %d to %d rows", rows, llava.get_input_embeddings().weight.shape[0])


@torch.no_grad()
def init_new_token_rows(llava, spec: TokenSpec, keep_seg: bool = False) -> None:
    """Set the rows of ``<SSEG>``, ``<REFMASK>`` and ``<REFBOX>`` to the mean of the pretrained rows.

    Both the input embedding and the output head (not tied in LLaVA-1.5) are initialised; the pretrained rows
    are those below the first added token (ids 0..31999 for LLaVA-1.5). ``keep_seg`` leaves the ``<SSEG>`` rows
    as loaded: a converted LISA checkpoint holds LISA's trained ``[SEG]`` rows there (``model.pretrained_seg_token``).
    """
    new_ids = [spec.seg_token_id, spec.ref_mask_token_id, spec.ref_box_token_id]
    num_pretrained = min(spec.image_token_id, *new_ids)
    if keep_seg:
        new_ids = new_ids[1:]
    for weight in (llava.get_input_embeddings().weight, llava.get_output_embeddings().weight):
        if weight.shape[0] <= max(new_ids):
            raise ValueError(f"embedding matrix has {weight.shape[0]} rows, but token id {max(new_ids)} is needed")
        weight[new_ids] = weight[:num_pretrained].float().mean(dim=0).to(weight.dtype)


def add_lora(llava, cfg: Config) -> int:
    """Inject LoRA (r, alpha, dropout from the config) into the language model's attention only.

    The pattern is anchored at ``language_model.layers`` because CLIP's attention also has ``q_proj`` and
    ``v_proj`` modules, which must stay untouched (LISA excludes them as well). Returns the module count.
    """
    from peft import LoraConfig, inject_adapter_in_model

    names = "|".join(re.escape(n) for n in cfg.model.lora_target_modules)
    lora_config = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
        target_modules=rf".*language_model\.layers\.\d+\.self_attn\.({names})",
    )
    inject_adapter_in_model(lora_config, llava)
    count = 0
    for name, module in llava.named_modules():
        if hasattr(module, "lora_A"):
            count += 1
            if "language_model" not in name:
                raise RuntimeError(f"LoRA was attached outside the language model: {name}")
    if count == 0:
        raise ValueError(f"no module of the language model matches lora_target_modules={cfg.model.lora_target_modules}")
    for name, param in llava.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
    return count


def make_embeddings_trainable(llava) -> None:
    """Input embeddings and output head become trainable fp32 copies (full-precision weights kept under bf16 autocast).

    Fig. 3 draws the MLLM as frozen apart from LoRA; training these two matrices follows LISA (paper Sec. 4.1)
    and is what lets the new ``<SSEG>`` row (Sec. 3.1) be learned. All rows are trainable, not only the new ones.
    """
    for module in (llava.get_input_embeddings(), llava.get_output_embeddings()):
        module.float()
        module.requires_grad_(True)


def image_features(llava, images_clip: torch.Tensor) -> torch.Tensor:
    """Projected CLIP patch features ``(B, image_seq_len, hidden)``; frozen, so no graph is built."""
    with torch.no_grad():
        pixels = images_clip.to(llava.model.vision_tower.dtype)
        features = llava.model.get_image_features(pixels).pooler_output
        return torch.stack(list(features))


def clip_cls_features(llava, pixels: torch.Tensor) -> torch.Tensor:
    """CLS token of CLIP's penultimate layer for anchor crops ``(R, 3, C, C)`` -> ``(R, clip_hidden)`` float32.

    This is the anchor mask embedding input of SegLLM: the layer is the one LLaVA
    uses for patches (``vision_feature_layer``), before the post layer norm.
    """
    with torch.no_grad():
        tower = llava.model.vision_tower
        out = tower(pixel_values=pixels.to(tower.dtype), output_hidden_states=True)
        return out.hidden_states[llava.config.vision_feature_layer][:, 0].float()
