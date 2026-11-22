"""Real-weights test of the CORA network on a GPU: LLaVA-1.5-7B (bf16) + SAM ViT-H on real Cityscapes frames.

Run with ``CORA_TEST_GPU=1`` on a machine with one GPU. It builds the model, runs optimizer steps at the
training batch geometry (2 images x 3 sequences, gradient checkpointing), the K-1 branch (2 images x 6
rephrasings, no gradient) and ``generate()``, checks that losses and gradients are finite, and writes the
timing and memory numbers to ``_out/model_real.json`` (printed with ``pytest -s``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from cora import constants as C
from cora.config import Config
from cora.model.build import build_model, build_tokenizer
from cora.model.teacher import Teacher
from tests.fixtures import ImageInput, make_batch

cv2 = pytest.importorskip("cv2")

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SAM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
LABEL_NAMES = {7: "road", 8: "sidewalk", 11: "building", 21: "vegetation", 23: "sky", 24: "person", 26: "car"}


def sam_image(image: np.ndarray, size: int = 1024):
    """SAM preprocessing: resize the longest side, normalise on 0-255 values, zero pad bottom/right."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = np.zeros((size, size, 3), dtype=np.float32)
    out[:nh, :nw] = (resized - SAM_MEAN) / SAM_STD
    return torch.from_numpy(out).permute(2, 0, 1), (nh, nw)


def clip_image(image: np.ndarray, size: int = 336, pad_value: np.ndarray = CLIP_MEAN * 255) -> torch.Tensor:
    """CLIP view: pad to a square with the CLIP mean colour, resize, normalise."""
    h, w = image.shape[:2]
    side = max(h, w)
    canvas = np.empty((side, side, 3), dtype=np.float32)
    canvas[:] = pad_value
    canvas[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = image
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC) / 255.0
    return torch.from_numpy(((resized - CLIP_MEAN) / CLIP_STD).astype(np.float32)).permute(2, 0, 1)


def anchor_inputs(image: np.ndarray, mask: np.ndarray, size: int = 336):
    """SegLLM anchor: zero outside the mask, tight crop, zero pad to a square (top-left), CLIP view + SAM-frame box."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = (image * mask[..., None])[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    side = max(crop.shape[:2])
    square = np.zeros((side, side, 3), dtype=np.float32)
    square[: crop.shape[0], : crop.shape[1]] = crop
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_CUBIC) / 255.0
    pixels = torch.from_numpy(((resized - CLIP_MEAN) / CLIP_STD).astype(np.float32)).permute(2, 0, 1)
    scale = 1024 / max(image.shape[:2])
    box = torch.tensor([y0, x0, y1, x1], dtype=torch.float32) * scale / 1024
    return pixels, box


def load_frame(root: Path, image_path: Path):
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    label_path = root / "gtFine" / image_path.relative_to(root / "leftImg8bit").parent / (
        image_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    )
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    assert label is not None and label.shape == image.shape[:2], label_path
    return image, label


def top_classes(label: np.ndarray, count: int):
    ids, counts = np.unique(label[label > 0], return_counts=True)
    order = np.argsort(-counts)
    return [int(ids[i]) for i in order[:count]]


def build_real_batch(root: Path, cfg, tokenizer, spec, num_images: int = 2, kind: str = "labeled", **kwargs):
    paths = sorted((root / "leftImg8bit" / "train").glob("*/*_leftImg8bit.png"))[:num_images]
    assert len(paths) == num_images
    inputs, masks, queries, references = [], [], [], []
    for path in paths:
        image, label = load_frame(root, path)
        a, b = top_classes(label, 2)
        sam, resize = sam_image(image)
        inputs.append(ImageInput(sam, clip_image(image), resize, image.shape[:2], path.stem))
        name_a, name_b = LABEL_NAMES.get(a, f"label {a}"), LABEL_NAMES.get(b, f"label {b}")
        queries += [
            f"Segment the {name_a}",
            f"Segment the {name_b} that fills a large part of the scene",
            f"Segment the {name_a} next to {C.REF_MASK_TOKEN} {C.REF_BOX_TOKEN}",
        ]
        mask_a, mask_b = (label == a).astype(np.uint8), (label == b).astype(np.uint8)
        masks += [torch.from_numpy(mask_a), torch.from_numpy(mask_b), torch.from_numpy(mask_a)]
        references.append(anchor_inputs(image, mask_b.astype(bool)))
    return make_batch(
        cfg,
        tokenizer,
        spec,
        kind=kind,
        seqs_per_image=[3] * num_images,
        queries=queries,
        images=inputs,
        masks=masks,
        references=references,
        **kwargs,
    )


def mask_loss(out, batch) -> torch.Tensor:
    losses = []
    for logits, target in zip(out.mask_logits, batch.masks):
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits, (target == 1).to(logits.dtype)))
    return torch.stack(losses).mean()


def timed(fn, repeat: int):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) / repeat


@pytest.mark.gpu
@pytest.mark.data
def test_real_llava_sam_forward_backward_and_generate(cityscapes_root):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cfg = Config()  # real LLaVA-1.5-7B, SAM ViT-H, bf16, gradient checkpointing
    assert cfg.model.precision == "bf16" and cfg.model.gradient_checkpointing and not cfg.model.tiny
    report: dict = {}

    start = time.perf_counter()
    tokenizer, spec = build_tokenizer(cfg)
    assert (spec.image_seq_len, spec.clip_image_size) == (576, 336)
    model = build_model(cfg, spec).cuda()
    model.attach_tokenizer(tokenizer)
    torch.cuda.synchronize()
    report["load_seconds"] = round(time.perf_counter() - start, 1)
    report["load_memory_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    report["trainable_parameters_m"] = round(sum(p.numel() for p in trainable.values()) / 1e6, 1)
    assert sum("lora_A" in n for n in trainable) == 64  # q_proj and v_proj of 32 layers
    assert {"llava.model.language_model.embed_tokens.weight", "llava.lm_head.weight"} <= set(trainable)
    assert all(p.dtype == torch.float32 for n, p in trainable.items())
    assert model.llava.model.vision_tower.dtype == torch.bfloat16
    assert not any(p.requires_grad for p in model.llava.model.vision_tower.parameters())
    assert not any(p.requires_grad for p in model.sam.image_encoder.parameters())
    assert model.llava.get_input_embeddings().weight.shape == (32064, 4096)
    mean_row = model.llava.get_input_embeddings().weight[: spec.image_token_id].float().mean(0)
    torch.testing.assert_close(model.llava.get_input_embeddings().weight[spec.seg_token_id].float(), mean_row, atol=1e-5, rtol=1e-4)

    batch = build_real_batch(cityscapes_root, cfg, tokenizer, spec, num_images=2).to("cuda")
    assert batch.num_sequences == 6 and batch.ref_seq.tolist() == [2, 5]
    report["sequence_length"] = int(batch.input_ids.shape[1])

    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-6, weight_decay=0.0)
    model.train()

    def step():
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = out.ce_loss + mask_loss(out, batch)
        loss.backward()
        optimizer.step()
        return out, loss

    out, loss = step()  # first step: finite losses and gradients on every trainable tensor
    assert torch.isfinite(loss) and torch.isfinite(out.ce_loss)
    assert [tuple(m.shape) for m in out.mask_logits] == [(1024, 2048)] * 6
    report["first_ce_loss"], report["first_mask_loss"] = round(float(out.ce_loss.detach()), 4), round(float(mask_loss(out, batch).detach()), 4)
    missing = [n for n, p in trainable.items() if p.grad is None]
    assert not missing, missing[:5]
    assert all(torch.isfinite(p.grad).all() for p in trainable.values())
    assert any(float(p.grad.abs().max()) > 0 for n, p in trainable.items() if n.startswith("anchors"))
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)

    torch.cuda.reset_peak_memory_stats()
    _, seconds = timed(step, repeat=3)
    report["train_step_seconds_2img_x_3seq"] = round(seconds, 2)
    report["train_step_peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    assert torch.isfinite(model(batch).ce_loss)

    # K-1 branch: 2 images x 6 rephrasings, online teacher, no gradient
    unlabeled = build_real_batch(
        cityscapes_root, cfg, tokenizer, spec, num_images=2, kind="unlabeled", aux_per_seq=2
    ).to("cuda")
    aux = unlabeled.aux
    teacher = Teacher(model, "online")
    teacher.start()
    sam_embeddings = model.encode_sam(aux.images_sam)
    torch.cuda.reset_peak_memory_stats()
    logits, seconds = timed(lambda: teacher.predict(aux, sam_embeddings), repeat=2)
    assert len(logits) == aux.num_sequences and all(torch.isfinite(m).all() for m in logits)
    report["teacher_sequences"] = aux.num_sequences
    report["teacher_seconds"] = round(seconds, 2)
    report["teacher_peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    assert model.training  # the training flag survives the teacher pass

    # frozen teacher on the real model: snapshot swap leaves the live weights alone
    frozen = Teacher(model, "frozen")
    frozen.start()
    before = {n: p.detach().clone() for n, p in list(trainable.items())[:20]}
    frozen.predict(aux, sam_embeddings)
    assert all(torch.equal(trainable[n].detach(), v) for n, v in before.items())
    del frozen

    # generate(): greedy text, then the mask of the first <SSEG>
    prompts = build_real_batch(cityscapes_root, cfg, tokenizer, spec, num_images=2, answer=None).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    (texts, masks), seconds = timed(lambda: model.generate(prompts, max_new_tokens=16), repeat=1)
    assert len(texts) == 6 and all(isinstance(t, str) for t in texts)
    assert all(m is None or tuple(m.shape) == (1024, 2048) for m in masks)
    report["generate_seconds_6_prompts_16_tokens"] = round(seconds, 2)
    report["generate_peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report["generated_text_example"] = texts[0]

    # force an <SSEG> into the answer to exercise the mask path with real shapes
    steps = []

    def scripted(module, args, output):
        token = [5000, spec.seg_token_id, spec.eos_token_id][min(len(steps), 2)]
        steps.append(token)
        fake = torch.zeros_like(output)
        fake[:, token] = 100.0
        return fake

    handle = model.llava.lm_head.register_forward_hook(scripted)
    texts, masks = model.generate(prompts, max_new_tokens=8)
    handle.remove()
    assert all(m is not None and tuple(m.shape) == (1024, 2048) and torch.isfinite(m).all() for m in masks)
    assert all(C.SEG_TOKEN in t for t in texts)

    # the trainable state is a plain dict of CPU tensors
    state = model.trainable_state_dict()
    assert set(state) == set(trainable) and all(v.device.type == "cpu" for v in state.values())
    report["trainable_state_gb"] = round(sum(v.numel() * v.element_size() for v in state.values()) / 2**30, 2)

    out_dir = Path("_out")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "model_real.json").write_text(json.dumps(report, indent=2))
    print("\nREAL MODEL REPORT " + json.dumps(report))
