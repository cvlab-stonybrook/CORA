#!/usr/bin/env python
"""Convert a released LISA checkpoint into the files CORA starts from.

    python tools/convert_lisa.py --lisa /path/to/LISA-7B-v1 --clip /path/to/clip-vit-large-patch14 \\
        --out pretrained/lisa-7b-v1

``--lisa`` is a local copy of ``xinlai/LISA-7B-v1`` (config.json, added_tokens.json, pytorch_model-*.bin and the
index), ``--clip`` a local copy of ``openai/clip-vit-large-patch14`` (config.json, model.safetensors): LISA's
checkpoint does not hold the CLIP weights. Both can be fetched with ``huggingface_hub.snapshot_download``.
Everything is converted and checked in memory before the first file is written; ``cora_init.json`` is removed at
the start of the writing and written last, so a folder that has it is complete.

The released LISA 7B checkpoints are built on LLaVA-Lightning-7B-v1-1, not on LLaVA-v1.5: Vicuna-7B v1.1, CLIP
ViT-L/14 at 224 px (256 image tokens), ONE linear projector, and ``<im_start>`` / ``<im_end>`` around the image.
The output folder holds

* ``llava/``: a ``LlavaForConditionalGeneration`` checkpoint (config.json, model.safetensors, tokenizer). The
  linear projector is written as ``linear_1`` = LISA's projector, identity activation, ``linear_2`` = identity
  matrix, which computes exactly LISA's projection. The tokenizer is CORA's (the LLaVA-1.5 tokenizer, same
  vocabulary below id 32000, plus ``<SSEG>``, ``<REFMASK>``, ``<REFBOX>``) with ``<im_start>`` and ``<im_end>``
  added; the embedding and output-head rows of ``<SSEG>``, ``<im_start>`` and ``<im_end>`` are LISA's rows of
  ``[SEG]``, ``<im_start>`` and ``<im_end>``. All other rows above 31999 are the mean of the vocabulary.
* ``sam_vit_h_lisa.pth``: LISA's SAM ViT-H (its trained mask decoder) in the format of ``segment_anything``.
* ``text_hidden_fcs.pt``: LISA's projection of the ``[SEG]`` hidden state to the SAM prompt.
* ``cora_init.json``: what was converted, and the configuration keys that use it.

A run uses it with (see configs/cityscapes/base.yaml)::

    model.llava=pretrained/lisa-7b-v1/llava  model.text_hidden_fcs_init=pretrained/lisa-7b-v1/text_hidden_fcs.pt
    model.pretrained_seg_token=true  paths.sam_checkpoint=pretrained/lisa-7b-v1/sam_vit_h_lisa.pth
    data.clip_preprocess=center_crop
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cora import constants as C  # noqa: E402
from cora.tokens import setup_tokenizer  # noqa: E402

LOG = logging.getLogger("convert_lisa")

EMBEDDING_ROWS = 32064  # as LLaVA-1.5: room for the added tokens without resizing
LISA_TOKENS = {"<im_start>": C.IM_START_TOKEN, "<im_end>": C.IM_END_TOKEN, "[SEG]": C.SEG_TOKEN}  # LISA token -> ours
SAM_PREFIX = "model.visual_model."
FCS_PREFIX = "model.text_hidden_fcs.0."
LLM_PREFIX = "model.layers."


def load_lisa_state(folder: Path) -> Dict[str, torch.Tensor]:
    index = json.loads((folder / "pytorch_model.bin.index.json").read_text())
    state: Dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        LOG.info("reading %s", folder / shard)
        state.update(torch.load(folder / shard, map_location="cpu", weights_only=True))
    missing = set(index["weight_map"]) - set(state)
    if missing:
        raise ValueError(f"{len(missing)} tensors of the index are not in the shards, first: {sorted(missing)[0]}")
    return state


def remap_rows(weight: torch.Tensor, lisa_ids: Dict[str, int], our_ids: Dict[str, int], pretrained: int) -> torch.Tensor:
    """Embedding or head matrix with CORA's row layout: the vocabulary, LISA's rows for its three tokens at our
    ids, the mean of the vocabulary elsewhere."""
    out = weight[:pretrained].float().mean(dim=0, keepdim=True).to(weight.dtype).repeat(EMBEDDING_ROWS, 1)
    out[:pretrained] = weight[:pretrained]
    for lisa_token, our_token in LISA_TOKENS.items():
        out[our_ids[our_token]] = weight[lisa_ids[lisa_token]]
    return out


def build_llava_config(lisa_config: dict, clip_dir: Path, image_token_id: int, pad_token_id: int):
    from transformers import CLIPConfig, LlamaConfig, LlavaConfig

    vision = CLIPConfig.from_pretrained(clip_dir).vision_config
    text = LlamaConfig(
        vocab_size=EMBEDDING_ROWS,
        hidden_size=lisa_config["hidden_size"],
        intermediate_size=lisa_config["intermediate_size"],
        num_hidden_layers=lisa_config["num_hidden_layers"],
        num_attention_heads=lisa_config["num_attention_heads"],
        num_key_value_heads=lisa_config["num_key_value_heads"],
        hidden_act=lisa_config["hidden_act"],
        max_position_embeddings=lisa_config["max_position_embeddings"],
        rms_norm_eps=lisa_config["rms_norm_eps"],
        bos_token_id=lisa_config["bos_token_id"],
        eos_token_id=lisa_config["eos_token_id"],
        pad_token_id=pad_token_id,
        tie_word_embeddings=False,
    )
    if lisa_config["mm_vision_select_layer"] != -2 or lisa_config["mm_vision_select_feature"] != "patch":
        raise ValueError("only LISA checkpoints that take CLIP's penultimate layer without CLS are supported")
    side = vision.image_size // vision.patch_size
    return LlavaConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=image_token_id,
        projector_hidden_act="linear",  # identity: with linear_2 = I the projector is LISA's single linear layer
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        image_seq_length=side * side,
        multimodal_projector_bias=True,
        tie_word_embeddings=False,
    )


def expected_tensors(config) -> Dict[str, torch.Size]:
    """Names and shapes of the model that will load the file, without allocating its weights."""
    from accelerate import init_empty_weights
    from transformers import LlavaForConditionalGeneration

    with init_empty_weights():
        model = LlavaForConditionalGeneration(config)
    return {name: tensor.shape for name, tensor in model.state_dict().items()}


def convert_llava(state: Dict[str, torch.Tensor], lisa_config: dict, lisa_tokens: Dict[str, int], clip_dir: Path,
                  tokenizer, spec, dtype: torch.dtype):
    from safetensors.torch import load_file

    our_ids = {C.SEG_TOKEN: spec.seg_token_id, C.IM_START_TOKEN: spec.im_start_token_id, C.IM_END_TOKEN: spec.im_end_token_id}
    pretrained = min(lisa_tokens.values())  # LISA's added tokens start right after the vocabulary
    if pretrained != spec.image_token_id or tokenizer.vocab_size != pretrained:
        raise ValueError(f"vocabulary sizes differ: LISA {pretrained}, CORA tokenizer {tokenizer.vocab_size} / {spec.image_token_id}")
    config = build_llava_config(lisa_config, clip_dir, spec.image_token_id, spec.pad_token_id)
    hidden = config.text_config.hidden_size
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        # rotary_emb.inv_freq is a buffer that the library of LISA's time stored; it is computed from the config now
        if name.startswith(LLM_PREFIX) and not name.endswith("rotary_emb.inv_freq"):
            out["model.language_model." + name[len("model."):]] = tensor
    out["model.language_model.norm.weight"] = state["model.norm.weight"]
    out["model.language_model.embed_tokens.weight"] = remap_rows(state["model.embed_tokens.weight"], lisa_tokens, our_ids, pretrained)
    out["lm_head.weight"] = remap_rows(state["lm_head.weight"], lisa_tokens, our_ids, pretrained)
    out["model.multi_modal_projector.linear_1.weight"] = state["model.mm_projector.weight"]
    out["model.multi_modal_projector.linear_1.bias"] = state["model.mm_projector.bias"]
    out["model.multi_modal_projector.linear_2.weight"] = torch.eye(hidden)
    out["model.multi_modal_projector.linear_2.bias"] = torch.zeros(hidden)
    expected = expected_tensors(config)
    # The vision tower's tensors are named "vision_model.<...>" in CLIP's checkpoint; inside LLaVA the library has
    # used both "vision_tower.vision_model.<...>" and "vision_tower.<...>": take the one this version expects.
    nested = any(name.startswith("model.vision_tower.vision_model.") for name in expected)
    clip = load_file(str(clip_dir / "model.safetensors"))
    for name, tensor in clip.items():
        if name.startswith("vision_model."):
            out["model.vision_tower." + (name if nested else name[len("vision_model."):])] = tensor
    out = {name: tensor.to(dtype).contiguous() for name, tensor in out.items()}
    for name in [n for n in out if n.endswith("position_ids") and n not in expected]:
        del out[name]  # an index buffer that newer versions of the library rebuild instead of storing
    missing, unexpected = sorted(set(expected) - set(out)), sorted(set(out) - set(expected))
    missing = [n for n in missing if not n.endswith("position_ids")]
    if missing or unexpected:
        raise ValueError(f"tensor names do not fit the model: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")
    wrong = [(n, tuple(out[n].shape), tuple(expected[n])) for n in out if out[n].shape != expected[n]]
    if wrong:
        raise ValueError(f"tensor shapes do not fit the model, first: {wrong[0]}")
    return config, out


def convert_sam(state: Dict[str, torch.Tensor], build_sam=None) -> Dict[str, torch.Tensor]:
    """LISA's SAM in the naming of ``segment_anything``; loaded once (strictly) into a SAM ViT-H as a check.
    ``build_sam`` replaces the ViT-H constructor in tests."""
    sam = {name[len(SAM_PREFIX):]: tensor.float() for name, tensor in state.items() if name.startswith(SAM_PREFIX)}
    if build_sam is None:
        from segment_anything import sam_model_registry

        build_sam = lambda: sam_model_registry["vit_h"](checkpoint=None)  # noqa: E731
    build_sam().load_state_dict(sam, strict=True)
    return sam


def convert_fcs(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    fcs = {name[len(FCS_PREFIX):]: tensor.float() for name, tensor in state.items() if name.startswith(FCS_PREFIX)}
    if sorted(fcs) != ["0.bias", "0.weight", "2.bias", "2.weight"]:
        raise ValueError(f"unexpected text_hidden_fcs tensors: {sorted(fcs)}")
    return fcs


def main(argv=None, build_sam=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lisa", type=Path, required=True, help="folder with the LISA checkpoint")
    parser.add_argument("--clip", type=Path, required=True, help="folder with openai/clip-vit-large-patch14")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--tokenizer", default="llava-hf/llava-1.5-7b-hf", help="tokenizer CORA uses (same vocabulary as LISA's)")
    parser.add_argument("--source-name", default="xinlai/LISA-7B-v1", help="name recorded in cora_init.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    lisa_config = json.loads((args.lisa / "config.json").read_text())
    lisa_tokens = json.loads((args.lisa / "added_tokens.json").read_text())
    absent = [t for t in LISA_TOKENS if t not in lisa_tokens]
    if absent:
        raise SystemExit(f"{args.lisa}/added_tokens.json lacks {absent}")
    if not lisa_config.get("mm_use_im_start_end"):
        raise SystemExit("this checkpoint was trained without <im_start>/<im_end>; the converter covers the released v1 models only")

    import shutil
    import tempfile

    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    # ---- everything is converted and checked before the output folder is touched ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="right")
    tokenizer.add_tokens([C.SEG_TOKEN, C.REF_MASK_TOKEN, C.REF_BOX_TOKEN, C.IM_START_TOKEN, C.IM_END_TOKEN], special_tokens=True)
    scratch = Path(tempfile.mkdtemp(prefix="cora_lisa_tokenizer_"))
    try:
        tokenizer.save_pretrained(scratch)
        tokenizer, spec = setup_tokenizer(str(scratch))  # the tokenizer exactly as a run will load it
        if not spec.lisa_layout or spec.vocab_size > EMBEDDING_ROWS:
            raise SystemExit("the saved tokenizer does not carry <im_start>/<im_end>, or has too many tokens")
        state = load_lisa_state(args.lisa)
        leftover = [n for n in state if not n.startswith((LLM_PREFIX, SAM_PREFIX, FCS_PREFIX))
                    and n not in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight",
                                  "model.mm_projector.weight", "model.mm_projector.bias")]
        if leftover:
            raise SystemExit(f"{len(leftover)} tensors of the checkpoint have no place in the converted model, first: {leftover[0]}")
        dtype = torch.bfloat16 if lisa_config.get("torch_dtype") == "bfloat16" else torch.float32
        config, llava = convert_llava(state, lisa_config, lisa_tokens, args.clip, tokenizer, spec, dtype)
        sam = convert_sam(state, build_sam)
        fcs = convert_fcs(state)
        side = config.vision_config.image_size // config.vision_config.patch_size

        # ---- writing: cora_init.json marks a complete folder, so it goes first out and last in --------
        llava_dir = args.out / "llava"
        llava_dir.mkdir(parents=True, exist_ok=True)
        (args.out / "cora_init.json").unlink(missing_ok=True)
        for item in scratch.iterdir():
            shutil.copy2(item, llava_dir / item.name)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    config.save_pretrained(llava_dir)
    LOG.info("writing %s (%d tensors)", llava_dir / "model.safetensors", len(llava))
    save_file(llava, str(llava_dir / "model.safetensors"), metadata={"format": "pt"})
    torch.save(sam, args.out / "sam_vit_h_lisa.pth")
    torch.save(fcs, args.out / "text_hidden_fcs.pt")
    info = {
        "source": args.source_name,
        "base": lisa_config.get("_name_or_path"),
        "clip": lisa_config.get("mm_vision_tower"),
        "clip_image_size": config.vision_config.image_size,
        "image_tokens": side * side,
        "rows": {C.SEG_TOKEN: [spec.seg_token_id, f"LISA [SEG] ({lisa_tokens['[SEG]']})"],
                 C.IM_START_TOKEN: [spec.im_start_token_id, f"LISA <im_start> ({lisa_tokens['<im_start>']})"],
                 C.IM_END_TOKEN: [spec.im_end_token_id, f"LISA <im_end> ({lisa_tokens['<im_end>']})"]},
        "use": {"model.llava": str(llava_dir), "model.text_hidden_fcs_init": str(args.out / "text_hidden_fcs.pt"),
                "model.pretrained_seg_token": True, "paths.sam_checkpoint": str(args.out / "sam_vit_h_lisa.pth"),
                "data.clip_preprocess": "center_crop"},
    }
    (args.out / "cora_init.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
