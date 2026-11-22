"""SAM side of CORA: the image encoder F_enc (Eq. 2) and the prompt-encoder / mask-decoder pair F_dec (Eq. 3).

SAM ViT-H is wired as in LISA (paper Sec. 4.1): the image encoder and the prompt
encoder are frozen, the mask decoder is trained, and the projected ``<SSEG>`` embedding is the only (sparse)
prompt token. The stock ``segment_anything.Sam.forward`` is ``no_grad``, so the three sub-modules are called
directly.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Tuple

import torch

from cora.config import Config

logger = logging.getLogger(__name__)

SAM_PATCH_SIZE = 16
SAM_PROMPT_DIM = 256


def tiny_sam(image_size: int, prompt_dim: int = SAM_PROMPT_DIM):
    """A random SAM with two transformer blocks, for unit tests (``image_size`` must be a multiple of 16)."""
    from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

    if image_size % SAM_PATCH_SIZE:
        raise ValueError(f"tiny SAM needs a data.sam_image_size that is a multiple of {SAM_PATCH_SIZE}, got {image_size}")
    grid = image_size // SAM_PATCH_SIZE
    depth = 2
    return Sam(
        image_encoder=ImageEncoderViT(
            depth=depth,
            embed_dim=32,
            img_size=image_size,
            mlp_ratio=2,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=2,
            patch_size=SAM_PATCH_SIZE,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[depth - 1],
            window_size=2,
            out_chans=prompt_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_dim,
            image_embedding_size=(grid, grid),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_dim, mlp_dim=64, num_heads=8),
            transformer_dim=prompt_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=64,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )


def build_sam(cfg: Config):
    """SAM with the pretrained weights of ``paths.sam_checkpoint`` (or a tiny random one for ``model.tiny``)."""
    if cfg.model.out_dim != SAM_PROMPT_DIM and not cfg.model.tiny:
        raise ValueError(f"model.out_dim must equal SAM's prompt width {SAM_PROMPT_DIM}, got {cfg.model.out_dim}")
    if cfg.model.tiny:
        return tiny_sam(cfg.data.sam_image_size, cfg.model.out_dim)
    from segment_anything import sam_model_registry

    if cfg.model.sam_type not in sam_model_registry:
        raise ValueError(f"unknown model.sam_type {cfg.model.sam_type!r}; valid: {sorted(sam_model_registry)}")
    path = cfg.resolve(cfg.paths.sam_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {path}")
    sam = sam_model_registry[cfg.model.sam_type](checkpoint=str(path))
    if sam.image_encoder.img_size != cfg.data.sam_image_size:
        raise ValueError(
            f"data.sam_image_size={cfg.data.sam_image_size} but SAM {cfg.model.sam_type} expects {sam.image_encoder.img_size}"
        )
    return sam


def configure_sam_trainable(sam, train_mask_decoder: bool) -> None:
    """Freeze SAM except the mask decoder (Fig. 3).

    The decoder's IoU head stays frozen: with ``multimask_output=False`` it receives no loss and no gradient.
    """
    sam.requires_grad_(False)
    if train_mask_decoder:
        for name, param in sam.mask_decoder.named_parameters():
            param.requires_grad_(not name.startswith("iou_prediction_head"))


@torch.no_grad()
def encode_images(sam, images: torch.Tensor) -> torch.Tensor:
    """Eq. 2: frozen SAM image embeddings ``(B, 256, h, w)`` in float32, one image at a time (LISA).

    ``images`` are SAM-preprocessed ``(B, 3, S, S)`` (normalised, zero padded). The caller sets the autocast
    context; the neck's layer norm returns float32 either way.
    """
    if images.dim() != 4 or images.shape[-1] != sam.image_encoder.img_size or images.shape[-2] != sam.image_encoder.img_size:
        raise ValueError(
            f"SAM expects images of shape (B, 3, {sam.image_encoder.img_size}, {sam.image_encoder.img_size}), "
            f"got {tuple(images.shape)}"
        )
    return torch.cat([sam.image_encoder(images[i : i + 1]).float() for i in range(images.shape[0])], dim=0)


def decode_masks(
    sam,
    image_embedding: torch.Tensor,
    prompts: torch.Tensor,
    resize_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
) -> torch.Tensor:
    """Eq. 3 for one image: ``(n, out_dim)`` <SSEG> prompts -> ``(n, H, W)`` float32 logits at original size.

    LISA recipe: the prompt is the single sparse token, the dense prompt is SAM's "no mask" embedding,
    ``multimask_output=False``, low-resolution logits are upsampled, cropped to the resized image and resized
    to the original resolution. Runs in float32 (autocast off) because the decoder is trained.
    """
    n = prompts.shape[0]
    with torch.autocast(device_type=prompts.device.type, enabled=False):
        sparse = prompts.float().unsqueeze(1)
        image_pe = sam.prompt_encoder.get_dense_pe()
        _, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
        low_res, _ = sam.mask_decoder(
            image_embeddings=image_embedding.float(),
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense.expand(n, -1, -1, -1),
            multimask_output=False,
        )
        masks = sam.postprocess_masks(low_res, input_size=tuple(resize_hw), original_size=tuple(orig_hw))
    return masks[:, 0].float()
