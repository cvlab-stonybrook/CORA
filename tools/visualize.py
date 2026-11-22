#!/usr/bin/env python
"""Qualitative results as in main Fig. 6 and supplement Fig. 1: one PNG panel per evaluation sample.

A panel shows, side by side, the image with the reference object (anchor, paper ``<REF>``) in green and the
ground-truth target in red, and the same image with the model's prediction in red; the query text stands underneath
(``<REF>`` in green). Samples come from the validation instruction sets of the configured dataset (Sec. 3.4.2); the
masks are predicted either teacher-forced (the answer containing ``<SSEG>`` is given, as in the evaluation of Tables 1
and 2) or generated (the model writes the answer and a mask is decoded from the ``<SSEG>`` token it emits).

    python tools/visualize.py --config configs/cityscapes/1_16.yaml --sets conditional --num-samples 8
    python tools/visualize.py --config configs/cityscapes/1_16.yaml --mode generate --indices 3 17 42
    python tools/visualize.py --config configs/cityscapes/1_16.yaml --checkpoint outputs/cityscapes_1_16/stage3/final.pt --out-dir /tmp/panels

Free-form queries (no stored instruction record, main Fig. 1, Fig. 6 and Fig. 7) are drawn with ``--query`` / ``--demo``:

    python tools/visualize.py --config configs/cityscapes/1_30.yaml --demo fig1 --image-id val/frankfurt/frankfurt_000001_064305
    python tools/visualize.py --config configs/cityscapes/1_30.yaml --demo fig7 --mode generate
    python tools/visualize.py --config configs/cityscapes/1_16.yaml --image-id <id> --anchor 13_2 \
        --query "Segment the individual who is riding <REF>"

``--demo fig1`` are the two probes of Fig. 1, ``--demo fig7`` the two failure queries of Fig. 7 (compound verb,
compositional spatial relation). A query with ``<REF>`` needs ``--anchor``, the region id of the anchor object
(``<class id>_<n>``, supplement Fig. 3; the region ids of an image are in its objects file written by
``tools/build_objects.py``). ``--target``
is an optional ground-truth region id drawn in red next to the prediction; without it the left image shows only the
anchor. ``--queries-file`` reads a JSON list of ``{"query", "image_id", "anchor", "target"}`` objects (the last three are
optional) for one query per image. The image is ``--image-id`` or, when neither it nor the file names one, the first image
of the split. Paired panels of two models (for example the 1/30 labeled-only and the 1/30 CORA run of Fig. 1) come from two
calls with different ``--checkpoint`` and ``--out-dir``.

``--num-samples`` is the number of samples per chosen set, drawn with ``--seed``; ``--indices`` selects dataset records
directly (give key=value overrides before --sets / --indices, which take several values). By default the weights of the last
finished stage of the run are used. Output: ``<out-dir>/NNN_<set>_<image>.png``
and ``index.json`` (file, record id, set, query, IoU of the sample and, in generate mode, the generated text).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cora import constants as C  # noqa: E402
from cora.config import Config, load_config  # noqa: E402
from cora.data.batch import Conversation, Sample  # noqa: E402
from cora.data.schema import ObjectRef  # noqa: E402

LOG = logging.getLogger("visualize")

RED = (255, 0, 0)  # target and prediction (Fig. 6)
GREEN = (0, 200, 0)  # reference object <REF> (Fig. 6)
TEXT_GREEN = (0, 140, 0)
TEXT_BLACK = (20, 20, 20)
TEXT_GRAY = (110, 110, 110)
GAP = 12  # pixels between and around the two images

# Queries of main Fig. 1 and Fig. 7, written as in the paper.
FIG1_QUERIES = (
    "Segment the rightmost four wheeler to transport people",
    "Segment the surface that a car can drive on",
)
FIG7_QUERIES = (
    "Segment the person who is sitting on and riding the bicycle",
    "Segment the person on the road and behind a truck",
)
DEMO_QUERIES = {"fig1": FIG1_QUERIES, "fig7": FIG7_QUERIES}


# ----------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------
def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45,
                 thickness: int = 2) -> np.ndarray:
    """Copy of ``image`` (H, W, 3 uint8) with the pixels of ``mask`` tinted with ``color`` and outlined in it."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got {image.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask {mask.shape} and image {image.shape[:2]} differ in size")
    inside = mask.astype(bool)
    out = image.astype(np.float32)
    out[inside] = (1.0 - alpha) * out[inside] + alpha * np.asarray(color, dtype=np.float32)
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    if inside.any() and thickness > 0:
        contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _font(size: int) -> ImageFont.ImageFont:
    """Pillow's built-in scalable font; the bitmap font of old Pillow versions ignores ``size``."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap_words(words: Sequence[str], font: ImageFont.ImageFont, max_width: int) -> List[List[str]]:
    """Greedy line breaking of ``words`` so that every line is at most ``max_width`` pixels wide (a long word gets its own line)."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: List[List[str]] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and probe.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(current)
    return lines


def _resize(array: np.ndarray, width: int, interpolation: int) -> np.ndarray:
    height = max(1, int(round(array.shape[0] * width / array.shape[1])))
    return cv2.resize(array, (width, height), interpolation=interpolation)


def render_panel(image: np.ndarray, query: str, target: Optional[np.ndarray], prediction: Optional[np.ndarray],
                 anchor: Optional[np.ndarray] = None, title: str = "", panel_width: int = 640) -> np.ndarray:
    """One qualitative panel as an (H, W, 3) uint8 array (Fig. 6 layout).

    Left: ``image`` with ``anchor`` in green and ``target`` in red (free-form queries may have no ``target``). Right: ``image`` with ``prediction`` in red; a
    ``prediction`` of ``None`` (no ``<SSEG>`` was generated) leaves the image untouched and says so. Underneath the
    ``title`` line (small, gray) and the ``query`` text, in which every word containing ``<REF>`` is green.
    """
    if panel_width < 64:
        raise ValueError(f"panel_width must be at least 64, got {panel_width}")
    image = np.ascontiguousarray(image)

    def fit(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        return _resize(mask.astype(np.uint8), panel_width, cv2.INTER_NEAREST) > 0

    base = _resize(image, panel_width, cv2.INTER_AREA)
    thickness = max(2, panel_width // 320)
    left = base
    anchor_fit = fit(anchor)
    if anchor_fit is not None:
        left = overlay_mask(left, anchor_fit, GREEN, thickness=thickness)
    target_fit = fit(target)
    if target_fit is not None:
        left = overlay_mask(left, target_fit, RED, thickness=thickness)
    prediction_fit = fit(prediction)
    right = base if prediction_fit is None else overlay_mask(base, prediction_fit, RED, thickness=thickness)

    width = 2 * panel_width + 3 * GAP
    small, large = max(11, panel_width // 46), max(14, panel_width // 30)
    small_font, large_font = _font(small), _font(large)
    lines = wrap_words(query.split(), large_font, width - 2 * GAP)
    line_height, small_height = int(large * 1.35), int(small * 1.5)
    height = base.shape[0] + 3 * GAP + 2 * small_height + len(lines) * line_height + GAP

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(Image.fromarray(left), (GAP, GAP))
    canvas.paste(Image.fromarray(right), (2 * GAP + panel_width, GAP))
    draw = ImageDraw.Draw(canvas)
    y = GAP + base.shape[0] + 4
    left_caption = ", ".join(([] if target is None else ["ground truth (red)"]) + ([] if anchor is None else ["reference (green)"]))
    draw.text((GAP, y), left_caption or "input image", fill=TEXT_GRAY, font=small_font)
    caption = "prediction (red)" if prediction is not None else "no <SSEG> generated: no mask"
    draw.text((2 * GAP + panel_width, y), caption, fill=RED if prediction is None else TEXT_GRAY, font=small_font)
    y += small_height + GAP
    if title:
        draw.text((GAP, y), title, fill=TEXT_GRAY, font=small_font)
    y += small_height
    space = draw.textlength(" ", font=large_font)
    for line in lines:
        x = float(GAP)
        for word in line:
            draw.text((x, y), word, fill=TEXT_GREEN if C.REF_TOKEN in word else TEXT_BLACK, font=large_font)
            x += draw.textlength(word, font=large_font) + space
        y += line_height
    return np.asarray(canvas)


def mask_iou(prediction: Optional[np.ndarray], target: Optional[np.ndarray]) -> Optional[float]:
    """IoU of a predicted mask (``None`` counts as empty) and the target; two empty masks give 1; no target gives ``None``."""
    if target is None:
        return None
    pred = np.zeros(target.shape, dtype=bool) if prediction is None else prediction.astype(bool)
    union = np.logical_or(pred, target).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, target).sum() / union)


# ----------------------------------------------------------------------------------------------
# Samples, predictions, output
# ----------------------------------------------------------------------------------------------
@dataclass
class PanelInput:
    """What a panel needs besides the prediction: the original image, the target and the anchor (reference) mask."""

    image: np.ndarray  # (H, W, 3) uint8 RGB
    query: str
    target: Optional[np.ndarray]  # (H, W) bool; None for a free-form query without ground truth
    anchor: Optional[np.ndarray]  # (H, W) bool, conditional queries only
    set_name: str
    image_id: str
    record_id: Optional[str] = None


@dataclass
class Prediction:
    """Model output for one sample: a bool mask at the original size (``None``: no <SSEG> was generated) and the text."""

    mask: Optional[np.ndarray]
    text: Optional[str] = None


class PanelSource(Protocol):
    """Anything that yields model-ready samples and the drawing inputs of the same record."""

    def __len__(self) -> int: ...

    def sample(self, index: int) -> Sample: ...

    def panel_input(self, index: int) -> PanelInput: ...


class EvalPanelSource:
    """Records of an :class:`cora.data.datasets.EvalDataset` (validation instruction sets, Sec. 4.1)."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def set_names(self) -> List[str]:
        return [record.set for record in self.dataset.records]

    def sample(self, index: int) -> Sample:
        return self.dataset[index]

    def panel_input(self, index: int) -> PanelInput:
        from cora.data.datasets import ImageContext

        record = self.dataset.records[index]
        context = ImageContext(self.dataset.cfg, self.dataset.adapter, self.dataset.split, record.image_id)
        anchor = context.mask(record.anchor) if record.anchor is not None else None
        return PanelInput(context.image, record.query, context.mask(record.target), anchor, record.set,
                          record.image_id, record.id)


def select_indices(set_names: Sequence[str], sets: Sequence[str], per_set: int, seed: int = 0) -> List[int]:
    """``per_set`` record indices of every set in ``sets`` (fewer if the set is smaller), drawn with ``seed``.

    ``set_names[i]`` is the set of record ``i``. The result lists the sets in the given order, indices ascending
    within a set. Raises when a requested set has no record.
    """
    if per_set < 1:
        raise ValueError(f"the number of samples per set must be at least 1, got {per_set}")
    chosen: List[int] = []
    for name in dict.fromkeys(sets):
        pool = [i for i, s in enumerate(set_names) if s == name]
        if not pool:
            raise ValueError(f"no evaluation record of the {name!r} set")
        chosen.extend(sorted(random.Random(f"{seed}|{name}").sample(pool, min(per_set, len(pool)))))
    return chosen


@dataclass
class FreeQuery:
    """A query typed by hand: text, image (``None``: the default image), anchor and target region ids (both optional)."""

    query: str
    image_id: Optional[str] = None
    anchor: Optional[str] = None  # region id "<class id>_<n>" of the <REF> object (supplement Fig. 3)
    target: Optional[str] = None  # region id of the ground truth, drawn in red when given


class QueryPanelSource:
    """Free-form queries on images of the configured dataset (main Fig. 1, Fig. 6, Fig. 7).

    A query without ``<REF>`` is a plain reasoning query; a query with ``<REF>`` needs an anchor region, which is passed to
    the model as mask crop and box (``model.reference.mode = tokens``) exactly as for the stored conditional records. The
    sample's target mask is all zeros, because the model's prediction does not depend on it.
    """

    SET_NAME = "freeform"

    def __init__(self, cfg: Config, spec, split: str, queries: Sequence[FreeQuery]) -> None:
        from cora.data.adapters import get_adapter

        if not queries:
            raise ValueError("no query given")
        self.cfg, self.spec, self.split = cfg, spec, split
        self.adapter = get_adapter(cfg)
        self.queries = list(queries)
        default_image = None
        for rank, item in enumerate(self.queries):
            self._check(rank, item)
            if item.image_id is None:
                if default_image is None:
                    ids = self.adapter.image_ids(split)
                    if not ids:
                        raise FileNotFoundError(f"the {split} split of {cfg.data.dataset} has no image")
                    default_image = ids[0]
                    LOG.info("no image id given for some queries: using %s", default_image)
                self.queries[rank] = FreeQuery(item.query, default_image, item.anchor, item.target)
        self._contexts: dict = {}

    def _check(self, rank: int, item: FreeQuery) -> None:
        tokens = (C.IMAGE_TOKEN, C.SEG_TOKEN, C.REF_MASK_TOKEN, C.REF_BOX_TOKEN)
        if not item.query.strip():
            raise ValueError(f"query {rank}: empty text")
        if any(token in item.query for token in tokens):
            raise ValueError(f"query {rank}: the text must not contain any of {tokens}")
        refs = item.query.count(C.REF_TOKEN)
        if refs > 1:
            raise ValueError(f"query {rank}: at most one {C.REF_TOKEN} is allowed, found {refs}")
        if refs and item.anchor is None:
            raise ValueError(f"query {rank} contains {C.REF_TOKEN} but gives no anchor region id (--anchor)")
        if item.anchor is not None and not refs:
            raise ValueError(f"query {rank} gives an anchor but its text has no {C.REF_TOKEN}")
        for rid in (item.anchor, item.target):
            if rid is not None:
                self._ref(rid)

    def _ref(self, region_id: str) -> ObjectRef:
        try:
            label_id, index = C.parse_region_id(region_id)
        except ValueError:
            raise ValueError(f"bad region id {region_id!r}; expected <class id>_<n>, for example 13_2") from None
        names = self.adapter.class_names
        if not 0 <= label_id < len(names) or index < 1:
            raise ValueError(f"bad region id {region_id!r}: class ids are 0..{len(names) - 1} and n starts at 1")
        return ObjectRef(label_id, str(names[label_id]), [region_id])

    def __len__(self) -> int:
        return len(self.queries)

    def _context(self, image_id: str):
        from cora.data.datasets import ImageContext

        if image_id not in self._contexts:
            self._contexts[image_id] = ImageContext(self.cfg, self.adapter, self.split, image_id)
        return self._contexts[image_id]

    def sample(self, index: int) -> Sample:
        from cora.data.datasets import EVAL_ANSWER

        item = self.queries[index]
        ctx = self._context(item.image_id)
        height, width = ctx.image.shape[:2]
        ref_pixels = ref_box = None
        if item.anchor is not None and self.cfg.model.reference.mode == "tokens":
            from cora.data.transforms import anchor_box, anchor_crop, clip_preprocess

            anchor = ctx.mask(self._ref(item.anchor))
            ref_pixels = clip_preprocess(anchor_crop(ctx.image, anchor), self.spec.clip_image_size, self.cfg.data.clip_preprocess)
            ref_box = anchor_box(anchor, self.cfg.data.sam_image_size)
        conversation = Conversation(item.query, EVAL_ANSWER, torch.zeros((height, width), dtype=torch.uint8), -1, self.SET_NAME,
                                    ref_pixels, ref_box, record_id=f"query{index}")
        return ctx.sample(self.spec, [conversation], "eval")

    def panel_input(self, index: int) -> PanelInput:
        item = self.queries[index]
        ctx = self._context(item.image_id)
        anchor = None if item.anchor is None else ctx.mask(self._ref(item.anchor))
        target = None if item.target is None else ctx.mask(self._ref(item.target))
        return PanelInput(ctx.image, item.query, target, anchor, self.SET_NAME, item.image_id, f"query{index}")


def load_queries_file(path: Path) -> List[FreeQuery]:
    """Read ``--queries-file``: a JSON list of objects with ``query`` and optional ``image_id``, ``anchor``, ``target``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must hold a non-empty JSON list of query objects")
    allowed = {"query", "image_id", "anchor", "target"}
    queries = []
    for rank, entry in enumerate(data):
        if not isinstance(entry, dict) or not isinstance(entry.get("query"), str) or set(entry) - allowed:
            raise ValueError(f"{path}: entry {rank} must be an object with a string 'query' and only the keys {sorted(allowed)}")
        queries.append(FreeQuery(**entry))
    return queries


def collect_queries(args: argparse.Namespace) -> List[FreeQuery]:
    """Free-form queries of the command line: ``--demo`` sets, ``--query`` texts and ``--queries-file`` (empty: none given)."""
    queries: List[FreeQuery] = []
    for name in args.demo or []:
        queries.extend(FreeQuery(text, args.image_id, None, args.target) for text in DEMO_QUERIES[name])
    queries.extend(FreeQuery(text, args.image_id, args.anchor, args.target) for text in args.query or [])
    if args.queries_file:
        queries.extend(load_queries_file(args.queries_file))
    if not queries and (args.anchor or args.target):
        raise ValueError("--anchor / --target need --query, --demo or --queries-file")
    return queries


@torch.no_grad()
def predict(model, collate: Callable, samples: Sequence[Sample], device: torch.device, mode: str = "teacher_forced",
            max_new_tokens: int = 32) -> List[Prediction]:
    """Masks (and, in generate mode, texts) of the model for ``samples``, which must hold one conversation each.

    ``collate`` is ``make_collate(tokenizer, spec, cfg, generation=mode == "generate")``. The masks are the logits
    above zero at the original image size, as in the evaluation of Sec. 4.1.
    """
    if mode not in ("teacher_forced", "generate"):
        raise ValueError(f"unknown mode {mode!r}; use teacher_forced or generate")
    was_training = model.training
    model.eval()
    try:
        batch = collate(list(samples)).to(device)
        if mode == "generate":
            texts, logits = model.generate(batch, max_new_tokens=max_new_tokens)
        else:
            texts, logits = [None] * batch.num_sequences, model(batch, compute_text_loss=False).mask_logits
    finally:
        model.train(was_training)
    return [Prediction(None if l is None else (l > 0).cpu().numpy(), t) for l, t in zip(logits, texts)]


def _file_stem(image_id: str) -> str:
    return image_id.replace("/", "_").replace("\\", "_")


def write_panels(inputs: Sequence[PanelInput], predictions: Sequence[Prediction], out_dir: Path,
                 panel_width: int = 640, mode: str = "teacher_forced") -> List[Path]:
    """Render and save one PNG per sample plus ``index.json``; returns the PNG paths in order."""
    if len(inputs) != len(predictions):
        raise ValueError(f"{len(inputs)} inputs but {len(predictions)} predictions")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    entries = []
    for rank, (item, pred) in enumerate(zip(inputs, predictions)):
        iou = mask_iou(pred.mask, item.target)
        title = f"{item.set_name} | {item.image_id} | {mode}" + ("" if iou is None else f" | IoU {iou:.2f}")
        panel = render_panel(item.image, item.query, item.target, pred.mask, item.anchor, title, panel_width)
        path = out_dir / f"{rank:03d}_{item.set_name}_{_file_stem(item.image_id)}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
            raise OSError(f"could not write {path}")
        paths.append(path)
        entries.append({"file": path.name, "record_id": item.record_id, "set": item.set_name, "image_id": item.image_id,
                        "query": item.query, "mode": mode, "iou": iou, "prediction_available": pred.mask is not None,
                        "generated_text": pred.text})
    (out_dir / "index.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    return paths


def visualize(model, collate: Callable, source: PanelSource, indices: Sequence[int], out_dir: Path,
              device: torch.device, mode: str = "teacher_forced", batch_size: int = 1, max_new_tokens: int = 32,
              panel_width: int = 640) -> List[Path]:
    """Predict the records ``indices`` of ``source`` and write their panels (Fig. 6 / supplement Fig. 1)."""
    if not indices:
        raise ValueError("no samples selected")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    for index in indices:
        if not 0 <= index < len(source):
            raise IndexError(f"sample index {index} is outside 0..{len(source) - 1}")
    predictions: List[Prediction] = []
    for start in range(0, len(indices), batch_size):
        samples = [source.sample(i) for i in indices[start:start + batch_size]]
        predictions.extend(predict(model, collate, samples, device, mode, max_new_tokens))
    inputs = [source.panel_input(i) for i in indices]
    return write_panels(inputs, predictions, out_dir, panel_width, mode)


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------
def default_checkpoint(cfg: Config) -> Path:
    """``final.pt`` of the last finished stage of the run."""
    from cora.engine import checkpoint as ckpt

    for stage in reversed(cfg.train.stages):
        final = ckpt.find_final(cfg.run_dir, stage.name)
        if final is not None:
            return final
    raise FileNotFoundError(f"no finished stage in {cfg.run_dir}; pass --checkpoint or --no-checkpoint")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML configuration file (defaults are used when omitted)")
    parser.add_argument("--checkpoint", help="final.pt or last.pt of a stage (default: last finished stage of the run)")
    parser.add_argument("--no-checkpoint", action="store_true", help="use the model before any CORA training")
    parser.add_argument("--sets", nargs="+", choices=C.LABELED_SETS,
                        help="instruction sets to draw samples from (default: the eval.sets that exist on disk)")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--query", action="append", help="free-form query text (repeatable); may contain <REF> together with --anchor")
    parser.add_argument("--demo", action="append", choices=sorted(DEMO_QUERIES),
                        help="add the queries of main Fig. 1 (fig1) or Fig. 7 (fig7); repeatable")
    parser.add_argument("--queries-file", help="JSON list of {query, image_id, anchor, target} objects")
    parser.add_argument("--image-id", help="image of the --query / --demo queries (default: first image of the split)")
    parser.add_argument("--anchor", help="region id <class id>_<n> of the <REF> object of the --query texts")
    parser.add_argument("--target", help="region id drawn as ground truth for the --query / --demo queries")
    parser.add_argument("--num-samples", type=int, default=4, help="samples per set (default 4)")
    parser.add_argument("--indices", type=int, nargs="+", help="record indices of the dataset to draw instead of random samples")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random choice of samples")
    parser.add_argument("--mode", choices=("teacher_forced", "generate"), help="default: eval.mode of the config")
    parser.add_argument("--max-new-tokens", type=int, help="default: eval.max_new_tokens of the config")
    parser.add_argument("--batch-size", type=int, help="default: eval.batch_size of the config")
    parser.add_argument("--panel-width", type=int, default=640, help="width in pixels of each of the two images (default 640)")
    parser.add_argument("--out-dir", help="output folder (default: <run_dir>/qualitative)")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides, e.g. data.partition=1_16")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from cora.data.collate import make_collate
    from cora.data.datasets import EvalDataset
    from cora.engine import checkpoint as ckpt
    from cora.model.build import build_model, build_tokenizer

    queries = collect_queries(args)
    if queries and (args.indices or args.sets):
        raise ValueError("--indices / --sets select stored records and cannot be combined with --query, --demo or --queries-file")
    cfg = load_config(args.config, args.overrides)
    mode = args.mode or cfg.eval.mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, spec = build_tokenizer(cfg)
    model = build_model(cfg, spec).to(device)
    if hasattr(model, "attach_tokenizer"):
        model.attach_tokenizer(tokenizer)
    if not args.no_checkpoint:
        path = Path(args.checkpoint) if args.checkpoint else default_checkpoint(cfg)
        model.load_trainable_state_dict(ckpt.load_checkpoint(path)["model"], strict=False)
        LOG.info("weights from %s", path)
    model.eval()

    if queries:
        source = QueryPanelSource(cfg, spec, args.split, queries)
        indices = list(range(len(source)))
    else:
        sets = args.sets or [s for s in cfg.eval.sets if cfg.instruction_files(args.split, s)]
        if not sets:
            raise FileNotFoundError(f"no {args.split} instruction files under {cfg.data_dir / 'instructions' / args.split}")
        source = EvalPanelSource(EvalDataset(cfg, spec, sets, split=args.split))
        indices = args.indices or select_indices(source.set_names, sets, args.num_samples, args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.run_dir / "qualitative"
    collate = make_collate(tokenizer, spec, cfg, generation=mode == "generate")
    paths = visualize(model, collate, source, indices, out_dir, device, mode,
                      args.batch_size or cfg.eval.batch_size,
                      args.max_new_tokens or cfg.eval.max_new_tokens, args.panel_width)
    print(f"{len(paths)} panels ({mode}) written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
