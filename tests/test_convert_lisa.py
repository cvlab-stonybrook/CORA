"""Tests of tools/convert_lisa.py on a small fake checkpoint in LISA's layout, and of the model built from it."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cora import constants as C
from cora.model.sam import tiny_sam
from cora.tokens import setup_tokenizer

REPO = Path(__file__).resolve().parents[1]
HIDDEN, CLIP_HIDDEN, OUT_DIM, SAM_SIZE = 32, 16, 256, 64
LISA_IDS = {"<im_patch>": 32000, "<im_start>": 32001, "<im_end>": 32002, "[SEG]": 32003}


def load_tool():
    spec = importlib.util.spec_from_file_location("tool_convert_lisa", REPO / "tools" / "convert_lisa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=["float32", "bfloat16"])
def converted(request, tmp_path_factory):
    """(tool, output folder, the fake LISA language model, its state dict). Every tensor differs from the value a
    freshly built model would have, so a tensor that is not carried over cannot go unnoticed; the released
    checkpoint stores its language model in bfloat16, which is the second variant."""
    from transformers import CLIPConfig, CLIPModel, LlamaConfig, LlamaForCausalLM

    dtype = torch.bfloat16 if request.param == "bfloat16" else torch.float32

    try:
        setup_tokenizer("llava-hf/llava-1.5-7b-hf")
    except OSError:
        pytest.skip("llava-hf/llava-1.5-7b-hf tokenizer is not in the local Hugging Face cache")
    root = tmp_path_factory.mktemp("lisa")
    lisa, clip_dir, out = root / "LISA", root / "clip", root / "out"
    lisa.mkdir()
    torch.manual_seed(0)
    llama = LlamaForCausalLM(LlamaConfig(
        vocab_size=32004, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=512, rms_norm_eps=1e-6, tie_word_embeddings=False)).eval()
    with torch.no_grad():
        for parameter in llama.parameters():  # also the norm weights, which are all ones in a new model
            parameter.add_(0.05 * torch.randn_like(parameter))
            parameter.copy_(parameter.to(dtype).float())  # values the checkpoint's dtype can hold
    state = {k: v.to(dtype) for k, v in llama.state_dict().items()}  # model.embed_tokens / model.layers / model.norm / lm_head: LISA's names
    state["model.mm_projector.weight"] = (0.5 * torch.randn(HIDDEN, CLIP_HIDDEN)).to(dtype)
    state["model.mm_projector.bias"] = torch.randn(HIDDEN).to(dtype)
    for name, shape in (("0.0.weight", (HIDDEN, HIDDEN)), ("0.0.bias", (HIDDEN,)), ("0.2.weight", (OUT_DIM, HIDDEN)), ("0.2.bias", (OUT_DIM,))):
        state[f"model.text_hidden_fcs.{name}"] = 0.3 * torch.randn(*shape)
    sam = tiny_sam(SAM_SIZE)
    with torch.no_grad():
        for parameter in sam.parameters():
            parameter.copy_(0.1 * torch.randn_like(parameter))
    for name, tensor in sam.state_dict().items():
        state[f"model.visual_model.{name}"] = tensor.clone()
    names = sorted(state)
    shards = {"pytorch_model-00001-of-00002.bin": names[::2], "pytorch_model-00002-of-00002.bin": names[1::2]}
    for shard, keys in shards.items():
        torch.save({k: state[k] for k in keys}, lisa / shard)
    (lisa / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {k: s for s, keys in shards.items() for k in keys}}))
    (lisa / "added_tokens.json").write_text(json.dumps(LISA_IDS))
    (lisa / "config.json").write_text(json.dumps({
        "_name_or_path": "./LLaVA/LLaVA-Lightning-7B-v1-1", "hidden_size": HIDDEN, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4, "hidden_act": "silu",
        "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "bos_token_id": 1, "eos_token_id": 2,
        "mm_use_im_start_end": True, "mm_vision_select_layer": -2, "mm_vision_select_feature": "patch",
        "mm_vision_tower": "openai/clip-vit-large-patch14", "torch_dtype": request.param}))
    clip = CLIPModel(CLIPConfig(
        vision_config=dict(hidden_size=CLIP_HIDDEN, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                           image_size=28, patch_size=14, projection_dim=8),
        text_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                         vocab_size=100, max_position_embeddings=16, projection_dim=8),
        projection_dim=8))
    with torch.no_grad():
        for parameter in clip.parameters():
            parameter.add_(0.2 * torch.randn_like(parameter))
            parameter.copy_(parameter.to(dtype).float())
    clip.save_pretrained(clip_dir)
    tool = load_tool()
    assert tool.main(["--lisa", str(lisa), "--clip", str(clip_dir), "--out", str(out)], build_sam=lambda: tiny_sam(SAM_SIZE)) == 0
    return SimpleNamespace(tool=tool, out=out, llama=llama, state=state, clip=clip, dtype=dtype, lisa=lisa, clip_dir=clip_dir)


def test_converted_llava_loads_completely_and_holds_lisas_rows(converted):
    from transformers import LlavaForConditionalGeneration

    tool, out, llama, state = converted.tool, converted.out, converted.llama, {k: v.float() for k, v in converted.state.items()}
    model, info = LlavaForConditionalGeneration.from_pretrained(out / "llava", output_loading_info=True)
    model = model.float()
    assert json.loads((out / "llava" / "config.json").read_text())["vision_feature_layer"] == -2
    vision = {k: v for k, v in model.model.vision_tower.state_dict().items() if not k.endswith("position_ids")}
    source = {k: v for k, v in converted.clip.vision_model.state_dict().items() if not k.endswith("position_ids")}
    assert len(vision) == len(source) and all(torch.equal(v, source[k.removeprefix("vision_model.")]) for k, v in vision.items())
    assert torch.equal(model.model.language_model.norm.weight, state["model.norm.weight"])
    assert not torch.equal(model.model.language_model.norm.weight, torch.ones(HIDDEN))
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"], info
    tokenizer, spec = setup_tokenizer(str(out / "llava"), image_seq_len=4, clip_image_size=28)
    assert spec.lisa_layout and spec.vocab_size <= tool.EMBEDDING_ROWS and int(model.config.image_token_id) == spec.image_token_id
    assert model.config.vision_config.image_size == 28 and model.config.image_seq_length == 4
    for ours, theirs in ((model.get_input_embeddings().weight, state["model.embed_tokens.weight"]),
                         (model.get_output_embeddings().weight, state["lm_head.weight"])):
        assert ours.shape == (tool.EMBEDDING_ROWS, HIDDEN) and torch.equal(ours[:32000], theirs[:32000])
        assert torch.equal(ours[spec.seg_token_id], theirs[LISA_IDS["[SEG]"]])
        assert torch.equal(ours[spec.im_start_token_id], theirs[LISA_IDS["<im_start>"]])
        assert torch.equal(ours[spec.im_end_token_id], theirs[LISA_IDS["<im_end>"]])
        mean = theirs[:32000].mean(dim=0)
        tolerance = 1e-6 if converted.dtype == torch.float32 else 1e-2  # the mean row is stored in the checkpoint's dtype
        assert torch.allclose(ours[spec.ref_mask_token_id], mean, atol=tolerance) and torch.allclose(ours[spec.image_token_id], mean, atol=tolerance)
    # the projector computes LISA's single linear layer
    features = torch.randn(3, 4, CLIP_HIDDEN)
    expected = features @ state["model.mm_projector.weight"].T + state["model.mm_projector.bias"]
    assert torch.allclose(model.model.multi_modal_projector(features), expected, atol=1e-5)
    # the language model is LISA's: same logits for the same text
    ids = torch.tensor([[1, 319, 13563, 1546, 263, 12758, 5199, LISA_IDS["[SEG]"]]])
    ours_ids = ids.clone()
    ours_ids[0, -1] = spec.seg_token_id
    with torch.no_grad():
        theirs = llama(input_ids=ids).logits
        ours = model.lm_head(model.model.language_model(input_ids=ours_ids).last_hidden_state)
    assert torch.allclose(ours[..., :32000], theirs[..., :32000], atol=1e-4)
    assert torch.allclose(ours[..., spec.seg_token_id], theirs[..., LISA_IDS["[SEG]"]], atol=1e-4)


def test_converted_sam_projection_and_description(converted):
    out, state = converted.out, converted.state
    exported = torch.load(out / "sam_vit_h_lisa.pth", weights_only=True)
    tiny_sam(SAM_SIZE).load_state_dict(exported, strict=True)
    expected = {k[len("model.visual_model."):]: v for k, v in state.items() if k.startswith("model.visual_model.")}
    assert set(exported) == set(expected) and all(torch.equal(exported[k], expected[k].float()) for k in expected)
    fcs = torch.nn.Sequential(torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, OUT_DIM), torch.nn.Dropout(0.0))
    fcs.load_state_dict(torch.load(out / "text_hidden_fcs.pt", weights_only=True))
    for ours, name in ((fcs[0].weight, "0.0.weight"), (fcs[0].bias, "0.0.bias"), (fcs[2].weight, "0.2.weight"), (fcs[2].bias, "0.2.bias")):
        assert torch.equal(ours, state[f"model.text_hidden_fcs.{name}"].float())
    info = json.loads((out / "cora_init.json").read_text())
    assert info["source"] == "xinlai/LISA-7B-v1" and info["image_tokens"] == 4 and info["use"]["model.pretrained_seg_token"] is True
    assert info["rows"][C.SEG_TOKEN][1].startswith("LISA [SEG]")


def lisa_forward(converted, batch, spec):
    """What LISA's code computes for the sequences of ``batch`` (model/LISA.py, llava_arch.py), written out with the
    checkpoint's own tensors: CLIP's penultimate layer without CLS -> the linear projector -> the embeddings between
    <im_start> and <im_end> -> the language model -> the last hidden state at the position before [SEG] ->
    text_hidden_fcs -> SAM's mask decoder with that single sparse prompt -> postprocess_masks."""
    from transformers import LlamaForCausalLM

    state = {k: v.float() for k, v in converted.state.items()}
    llama = LlamaForCausalLM(converted.llama.config).eval()
    llama.load_state_dict({k: v for k, v in state.items() if k.startswith(("model.layers", "model.embed", "model.norm", "lm_head"))})
    tower = converted.clip.vision_model.eval()  # the vision transformer of the CLIP model the converter read
    sam = tiny_sam(SAM_SIZE).eval()
    sam.load_state_dict({k[len("model.visual_model."):]: v for k, v in state.items() if k.startswith("model.visual_model.")})
    weight, bias = state["model.mm_projector.weight"], state["model.mm_projector.bias"]
    fcs = lambda h: torch.relu(h @ state["model.text_hidden_fcs.0.0.weight"].T + state["model.text_hidden_fcs.0.0.bias"]) \
        @ state["model.text_hidden_fcs.0.2.weight"].T + state["model.text_hidden_fcs.0.2.bias"]  # noqa: E731
    to_lisa = {spec.seg_token_id: LISA_IDS["[SEG]"], spec.im_start_token_id: LISA_IDS["<im_start>"], spec.im_end_token_id: LISA_IDS["<im_end>"]}
    out = []
    with torch.no_grad():
        for n in range(batch.input_ids.shape[0]):
            ids, image = batch.input_ids[n], int(batch.seq_image[n])
            features = tower(pixel_values=batch.images_clip[image : image + 1], output_hidden_states=True).hidden_states[-2][0, 1:] @ weight.T + bias
            embeddings = llama.get_input_embeddings()(torch.tensor([to_lisa.get(int(i), int(i)) for i in ids]))
            embeddings[ids == spec.image_token_id] = features
            length = int(batch.attention_mask[n].sum())
            hidden = llama.model(inputs_embeds=embeddings[None, :length]).last_hidden_state[0]
            before_seg = hidden[int((ids == spec.seg_token_id).nonzero()[0]) - 1]
            prompt = fcs(before_seg)
            _, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
            low, _ = sam.mask_decoder(image_embeddings=sam.image_encoder(batch.images_sam[image : image + 1]),
                                      image_pe=sam.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=prompt[None, None],
                                      dense_prompt_embeddings=dense, multimask_output=False)
            mask = sam.postprocess_masks(low, input_size=tuple(batch.resize_hw[image]), original_size=tuple(batch.orig_hw[image]))[0, 0]
            out.append((before_seg, prompt, mask))
    return out


def test_model_built_from_the_converted_files_computes_what_lisa_computes(converted, monkeypatch):
    """build_model with the configuration keys of cora_init.json: LISA's <SSEG> rows and projection are in place,
    the anchor tokens get the mean row, and the forward pass in LISA's prompt layout gives LISA's hidden state,
    SAM prompt and mask logits for the same input."""
    from cora.data.batch import Conversation, Sample
    from cora.data.collate import collate
    from cora.model import build as build_module
    from tests.fixtures import tiny_config

    out, state = converted.out, {k: v.float() for k, v in converted.state.items()}
    cfg = tiny_config()
    cfg.model.tiny = False  # the converted (small) checkpoint is loaded, not a random model
    cfg.model.llava = str(out / "llava")
    cfg.model.text_hidden_fcs_init = str(out / "text_hidden_fcs.pt")
    cfg.model.pretrained_seg_token = True
    cfg.data.sam_image_size = SAM_SIZE
    exported_sam = torch.load(out / "sam_vit_h_lisa.pth", weights_only=True)

    def small_sam(cfg):
        sam = tiny_sam(SAM_SIZE)
        sam.load_state_dict(exported_sam, strict=True)
        return sam

    monkeypatch.setattr(build_module, "build_sam", small_sam)
    tokenizer, spec = build_module.build_tokenizer(cfg)
    assert spec.lisa_layout and (spec.image_seq_len, spec.clip_image_size) == (4, 28)
    model = build_module.build_model(cfg, spec)
    model.attach_tokenizer(tokenizer)
    for ours, theirs in ((model.llava.get_input_embeddings().weight, state["model.embed_tokens.weight"]),
                         (model.llava.get_output_embeddings().weight, state["lm_head.weight"])):
        assert ours.dtype == torch.float32 and torch.equal(ours[spec.seg_token_id], theirs[LISA_IDS["[SEG]"]])
        assert torch.allclose(ours[spec.ref_box_token_id], theirs[:32000].mean(0), atol=1e-5)
        assert torch.allclose(ours[spec.ref_mask_token_id], theirs[:32000].mean(0), atol=1e-5)
    assert torch.equal(model.text_hidden_fcs[0].weight, state["model.text_hidden_fcs.0.0.weight"])
    samples = []
    torch.manual_seed(5)
    for i in range(2):
        mask = torch.zeros(40, 60, dtype=torch.uint8)
        mask[5:20, 10:30] = 1
        conversations = [Conversation(f"Please segment the thing number {j} in this image.", f"It is {C.SEG_TOKEN}.", mask, 0, "semantic")
                         for j in range(2)]
        samples.append(Sample(f"image{i}", torch.randn(3, SAM_SIZE, SAM_SIZE), torch.randn(3, 28, 28), (43, 64), (40, 60), conversations))
    batch = collate(samples, tokenizer, spec, cfg)
    assert (batch.input_ids == spec.im_start_token_id).sum(1).tolist() == [1] * 4
    assert (batch.input_ids == spec.im_end_token_id).sum(1).tolist() == [1] * 4
    model.eval()
    with torch.no_grad():
        result = model(batch)
    assert torch.isfinite(result.ce_loss) and len(result.mask_logits) == 4 and result.mask_logits[0].shape == (40, 60)
    for n, (hidden, prompt, mask) in enumerate(lisa_forward(converted, batch, spec)):
        assert torch.allclose(result.seg_hidden[n], hidden, atol=1e-4, rtol=1e-4), n
        assert torch.allclose(result.seg_embed[n], prompt, atol=1e-3, rtol=1e-4), n
        assert torch.allclose(result.mask_logits[n], mask, atol=1e-3, rtol=1e-3), n


def test_a_conversion_that_fails_writes_nothing_and_a_rerun_replaces_the_description(converted, tmp_path):
    tool = converted.tool
    # a checkpoint with a tensor that has no place in the converted model: stopped before any file is written
    broken = tmp_path / "LISA"
    broken.mkdir()
    for item in converted.lisa.iterdir():
        if item.name.endswith(".bin"):
            shard = torch.load(item, weights_only=True)
            shard["model.something_new.weight"] = torch.zeros(2)
            torch.save(shard, broken / item.name)
        else:
            (broken / item.name).write_bytes(item.read_bytes())
    index = json.loads((broken / "pytorch_model.bin.index.json").read_text())
    index["weight_map"]["model.something_new.weight"] = sorted(set(index["weight_map"].values()))[0]
    (broken / "pytorch_model.bin.index.json").write_text(json.dumps(index))
    fresh = tmp_path / "fresh"
    with pytest.raises(SystemExit, match="no place in the converted model"):
        tool.main(["--lisa", str(broken), "--clip", str(converted.clip_dir), "--out", str(fresh)], build_sam=lambda: tiny_sam(SAM_SIZE))
    assert not fresh.exists()
    # the same failure into a finished folder leaves that folder as it was
    before = {p.name: p.stat().st_mtime_ns for p in converted.out.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit):
        tool.main(["--lisa", str(broken), "--clip", str(converted.clip_dir), "--out", str(converted.out)], build_sam=lambda: tiny_sam(SAM_SIZE))
    assert {p.name: p.stat().st_mtime_ns for p in converted.out.rglob("*") if p.is_file()} == before
    assert "added_tokens.json" in tool.__doc__
