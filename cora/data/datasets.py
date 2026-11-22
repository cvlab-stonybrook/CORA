"""Datasets: everything between the files on disk and a :class:`cora.data.batch.Sample` (paper Sec. 3.2-3.4).

* :class:`LabeledDataset`: labeled images with the instruction sets of a training stage (Sec. 3.3, stages 1-3).
* :class:`UnlabeledDataset`: unlabeled images with pseudo-labels and class queries from the database D
  (Sec. 3.2.1, Fig. 3).
* :class:`EvalDataset`: one sample per stored instruction of a validation split (Sec. 4.1).

The two training datasets draw their randomness from a ``random.Random`` seeded with ``(seed, index)``, so a draw
does not depend on the number of loader workers or on which worker serves it (a stage resumed in the middle of a
pass hands its batches to other workers than the uninterrupted run did). Their ``index`` may be
any non-negative integer: :func:`cora.data.samplers.infinite_loader` passes ``epoch * len(dataset) + i`` so that
every draw has its own seed.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from cora import constants as C
from cora.config import Config, StageConfig
from cora.data import templates
from cora.data.adapters import DatasetAdapter, get_adapter
from cora.data.batch import Conversation, Sample
from cora.data.objects import ImageObjects, ensure_objects
from cora.data.query_db import QueryDB
from cora.data.schema import InstructionRecord, ObjectRef, iter_jsonl
from cora.data.transforms import anchor_box, anchor_crop, clip_preprocess, sam_preprocess
from cora.pseudo.io import has_prediction, prediction_paths, read_prediction
from cora.tokens import TokenSpec

LOG = logging.getLogger(__name__)

EVAL_ANSWER = f"It is {C.SEG_TOKEN}."  # teacher-forced answer of the evaluation samples
MAX_RESAMPLES = 64  # images tried by the unlabeled dataset before it gives up
STYLE_LLM = "llm"  # conditional records written by an LLM (Table 5, the provided instruction files) or the offline generator
STYLE_TEMPLATED = "templated"  # rule-based class-level conditional records (Fig. 2)
CONDITIONAL_STYLES = (STYLE_LLM, STYLE_TEMPLATED)
DEFAULT_STYLE_WEIGHTS: Mapping[str, float] = {STYLE_LLM: 1.0, STYLE_TEMPLATED: 1.0}  # 1:1
# data.conditional_sources -> how often each source of the conditional set is drawn
SOURCE_STYLE_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "both": DEFAULT_STYLE_WEIGHTS,
    "llm": {STYLE_LLM: 1.0, STYLE_TEMPLATED: 0.0},
    "templated": {STYLE_LLM: 0.0, STYLE_TEMPLATED: 1.0},
}


def stream_rng(seed: int, index: int) -> random.Random:
    """``random.Random`` seeded from ``(seed, index)``; identical across workers, processes and runs."""
    return random.Random(f"cora-data|{seed}|{index}")


def _check_index(index: int) -> int:
    if not isinstance(index, (int, np.integer)) or index < 0:
        raise IndexError(f"dataset index must be a non-negative integer, got {index!r}")
    return int(index)


def record_style(record: InstructionRecord) -> str:
    """``"templated"`` for rule-based class-level conditional records, ``"llm"`` for every other record."""
    return STYLE_TEMPLATED if record.set == C.SET_CONDITIONAL and record.meta.get("style") == STYLE_TEMPLATED else STYLE_LLM


def eval_group(record: InstructionRecord) -> str:
    """Group name an evaluation record is reported under (Sec. 4.1, Tables 1 and 4).

    The set name, except for the rule-based conditional set, which is reported as ``conditional_templated`` so
    that the paper-style (LLM-generated) ``conditional`` numbers are not blended with the added class-level set.
    """
    style = record_style(record)
    return record.set if style == STYLE_LLM else f"{record.set}_{style}"


def _read_label(pseudo_dir: Path, image_id: str) -> np.ndarray:
    """Pseudo-label map (H, W) uint8 only; the confidence PNG is not decoded."""
    path = prediction_paths(pseudo_dir, image_id)[0]
    if not path.is_file():
        raise FileNotFoundError(f"pseudo-label file not found: {path}")
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None or label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError(f"{path}: expected a single-channel 8-bit PNG")
    return label


def load_instruction_records(cfg: Config, set_name: str, split: str) -> List[InstructionRecord]:
    """Records of a set from every file ``cfg.instruction_files(split, set_name)`` (the set file and its variants).

    Raises ``FileNotFoundError`` naming the expected file when the set does not exist, and ``ValueError`` for
    records of another dataset / split / set or duplicate record ids.
    """
    files = cfg.instruction_files(split, set_name)
    if not files:
        raise FileNotFoundError(
            f"no instruction file for the {set_name!r} set of the {split} split: expected {cfg.instructions_path(split, set_name)}"
            " (put the provided instruction files there, or build the rule-based sets with tools/generate_rule_sets.py)"
        )
    records: List[InstructionRecord] = []
    seen: set = set()
    for path in files:
        for record in iter_jsonl(path):
            if (record.dataset, record.split, record.set) != (cfg.data.dataset, split, set_name):
                raise ValueError(
                    f"{path}: record {record.id} is {record.dataset}/{record.split}/{record.set}, "
                    f"expected {cfg.data.dataset}/{split}/{set_name}"
                )
            record.validate()
            if record.id in seen:
                raise ValueError(f"{path}: duplicate record id {record.id!r}")
            seen.add(record.id)
            records.append(record)
    if not records:
        raise ValueError(f"the instruction file(s) of the {set_name!r} set are empty: {[str(f) for f in files]}")
    return records


class ImageContext:
    """Lazily loaded image, label map and objects of one image; builds target and anchor masks of records."""

    def __init__(self, cfg: Config, adapter: DatasetAdapter, split: str, image_id: str) -> None:
        self.cfg, self.adapter, self.split, self.image_id = cfg, adapter, split, image_id
        self.image = adapter.load_image(image_id)
        self._label: Optional[np.ndarray] = None
        self._objects: Optional[ImageObjects] = None

    @property
    def label(self) -> np.ndarray:
        if self._label is None:
            self._label = self.adapter.load_label(self.image_id)
            if self._label.shape != self.image.shape[:2]:
                raise ValueError(f"{self.image_id}: label shape {self._label.shape} differs from image shape {self.image.shape[:2]}")
        return self._label

    @property
    def objects(self) -> ImageObjects:
        if self._objects is None:
            self._objects = ensure_objects(self.cfg, self.adapter, self.split, self.image_id)
            if (self._objects.height, self._objects.width) != self.image.shape[:2]:
                raise ValueError(f"{self.image_id}: stored objects are {self._objects.height}x{self._objects.width} but the image is {self.image.shape[:2]}")
        return self._objects

    def mask(self, ref: ObjectRef) -> np.ndarray:
        """(H, W) bool mask of a target or anchor: its regions, or every pixel of the class when it has none."""
        if ref.region_ids:
            mask = self.objects.mask(ref.region_ids)
        else:
            mask = self.label == ref.label_id
        if not mask.any():
            raise ValueError(f"{self.image_id}: the mask of {ref.label_name!r} {ref.region_ids or '(whole class)'} is empty")
        return mask

    def sample(self, spec: TokenSpec, conversations: List[Conversation], kind: str) -> Sample:
        """Preprocess the image for SAM and CLIP and wrap it with its conversations."""
        sam, resize_hw = sam_preprocess(self.image, self.cfg.data.sam_image_size)
        clip = clip_preprocess(self.image, spec.clip_image_size, self.cfg.data.clip_preprocess)
        height, width = self.image.shape[:2]
        return Sample(self.image_id, sam, clip, resize_hw, (height, width), conversations, kind)


def record_conversation(
    ctx: ImageContext, spec: TokenSpec, record: InstructionRecord, query: str, answer: str, ignore_void: bool = False,
    set_name: Optional[str] = None,
) -> Conversation:
    """Conversation of a stored instruction: target mask and, for conditional records, the anchor inputs.

    ``ref_pixels`` / ``ref_box`` are made only when ``cfg.model.reference.mode == "tokens"`` (Sec. 3.3).
    ``set_name`` (default: the record's set) is the group the evaluator reports the sample under.
    """
    cfg = ctx.cfg
    target = ctx.mask(record.target)
    mask = target.astype(np.uint8)
    if ignore_void:
        mask[ctx.label == C.IGNORE_LABEL] = C.IGNORE_LABEL
    ref_pixels = ref_box = None
    if record.set == C.SET_CONDITIONAL and cfg.model.reference.mode == "tokens":
        anchor = ctx.mask(record.anchor)
        ref_pixels = clip_preprocess(anchor_crop(ctx.image, anchor), spec.clip_image_size, cfg.data.clip_preprocess)
        ref_box = anchor_box(anchor, cfg.data.sam_image_size)
    return Conversation(
        query=query, answer=answer, mask=torch.from_numpy(mask), class_id=int(record.target.label_id),
        set_name=record.set if set_name is None else set_name, ref_pixels=ref_pixels, ref_box=ref_box, record_id=record.id,
    )


# ----------------------------------------------------------------------------------------------
# Labeled images
# ----------------------------------------------------------------------------------------------
class LabeledDataset(Dataset):
    """Labeled images of ``cfg.data.partition`` with the instruction sets of a stage (Sec. 3.3, Eq. 4-5).

    Each call draws an instruction set of the stage (``stage.set_weights``), then a labeled image that has
    records in that set, then up to ``cfg.data.queries_per_image`` distinct records of the image (for the
    semantic set: distinct classes, each with a freshly drawn template). The ``index`` only seeds the draw.

    The conditional set has two sources: LLM-generated records (``"llm"``) and rule-based
    class-level records (``"templated"``). ``cfg.data.conditional_sources`` chooses them ("both": 1:1,
    independent of how many records each source holds per image; "llm" or "templated": one source only);
    ``conditional_style_weights`` overrides it with explicit weights (a weight of 0 leaves the source out).
    When an image has records of one source only, that source is used.
    """

    def __init__(
        self, cfg: Config, stage: StageConfig, spec: TokenSpec, seed: int = 0,
        conditional_style_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.cfg, self.stage, self.spec, self.seed = cfg, stage, spec, seed
        if conditional_style_weights is None:
            if cfg.data.conditional_sources not in SOURCE_STYLE_WEIGHTS:
                raise ValueError(f"data.conditional_sources must be one of {sorted(SOURCE_STYLE_WEIGHTS)}, "
                                 f"got {cfg.data.conditional_sources!r}")
            conditional_style_weights = SOURCE_STYLE_WEIGHTS[cfg.data.conditional_sources]
        style_weights = dict(conditional_style_weights)
        if (set(style_weights) - set(CONDITIONAL_STYLES) or any(w < 0 for w in style_weights.values())
                or not any(w > 0 for w in style_weights.values())):
            raise ValueError(f"conditional_style_weights {style_weights} must map {CONDITIONAL_STYLES} to non-negative "
                             "weights, at least one above zero")
        self.style_weights = {style: float(style_weights.get(style, 0.0)) for style in CONDITIONAL_STYLES}
        sets = list(stage.labeled_sets)
        if not sets:
            raise ValueError(f"stage {stage.name}: labeled_sets is empty")
        weights = list(stage.set_weights) if stage.set_weights is not None else [1.0] * len(sets)
        if len(weights) != len(sets) or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"stage {stage.name}: set_weights {weights} must be non-negative, sum above zero and match {sets}")
        self.sets, self.weights = sets, weights
        self.adapter = get_adapter(cfg)
        self.image_ids = self.adapter.partition(cfg.data.partition, "labeled")
        in_partition = set(self.image_ids)
        self._records: Dict[str, Dict[str, List[InstructionRecord]]] = {}
        for set_name in sets:
            by_image: Dict[str, List[InstructionRecord]] = {}
            for record in load_instruction_records(cfg, set_name, "train"):
                if set_name == C.SET_CONDITIONAL and self.style_weights[record_style(record)] <= 0:
                    continue
                if record.image_id in in_partition:
                    by_image.setdefault(record.image_id, []).append(record)
            if not by_image:
                files = [str(f) for f in cfg.instruction_files("train", set_name)]
                sources = (f" from the source(s) {[s for s, w in self.style_weights.items() if w > 0]} "
                           f"(data.conditional_sources)" if set_name == C.SET_CONDITIONAL else "")
                raise ValueError(f"the {set_name!r} set holds no record{sources} for the labeled images of partition "
                                 f"{cfg.data.partition}: {files}")
            self._records[set_name] = by_image
            if set_name == C.SET_CONDITIONAL:
                per_style = {style: len({i for i, rs in by_image.items() if any(record_style(r) == style for r in rs)})
                             for style in CONDITIONAL_STYLES}
                LOG.info("conditional set of stage %s: LLM-written records on %d and rule-based records on %d of the %d "
                         "labeled images (data.conditional_sources=%s)", stage.name, per_style[STYLE_LLM],
                         per_style[STYLE_TEMPLATED], len(self.image_ids), cfg.data.conditional_sources)
        self._images = {s: sorted(by_image) for s, by_image in self._records.items()}
        # Optional share of samples from the ReasonSeg train split (LISA's data mix has one in 16).
        self._reasonseg = None
        if cfg.data.reasonseg_train_share > 0:
            from cora.data.reasonseg import ReasonSegTrainSet

            self._reasonseg = ReasonSegTrainSet(cfg, spec)
            LOG.info("stage %s: %.1f%% of the labeled samples come from %d of the %d ReasonSeg train images",
                     stage.name, 100 * cfg.data.reasonseg_train_share, len(self._reasonseg), self._reasonseg.available)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _pick(self, set_name: str, records: List[InstructionRecord], rng: random.Random) -> List[InstructionRecord]:
        count = self.cfg.data.queries_per_image
        if set_name == C.SET_SEMANTIC:
            by_class: Dict[int, List[InstructionRecord]] = {}
            for record in records:
                by_class.setdefault(record.target.label_id, []).append(record)
            classes = rng.sample(sorted(by_class), min(count, len(by_class)))
            return [rng.choice(by_class[c]) for c in classes]
        if set_name == C.SET_CONDITIONAL:
            return self._pick_conditional(records, count, rng)
        return rng.sample(records, min(count, len(records)))

    def _pick_conditional(self, records: List[InstructionRecord], count: int, rng: random.Random) -> List[InstructionRecord]:
        """Distinct conditional records: for each one the source is drawn by ``style_weights``, then a record of it."""
        by_style: Dict[str, List[InstructionRecord]] = {}
        for record in records:
            by_style.setdefault(record_style(record), []).append(record)
        picked: List[InstructionRecord] = []
        for _ in range(min(count, len(records))):
            styles = [style for style, pool in by_style.items() if pool]
            style = rng.choices(styles, weights=[self.style_weights[s] for s in styles])[0]
            pool = by_style[style]
            picked.append(pool.pop(rng.randrange(len(pool))))
        return picked

    def _query(self, record: InstructionRecord, rng: random.Random) -> str:
        """Stored query; class-level semantic records get a fresh template and class phrase (Sec. 3.4.2)."""
        if record.set == C.SET_SEMANTIC and record.meta.get("class_level"):
            phrase = rng.choice(templates.class_phrases(self.cfg.data.dataset, record.target.label_id))
            return templates.semantic_query(phrase, rng)
        return record.query

    def __getitem__(self, index: int) -> Sample:
        rng = stream_rng(self.seed, _check_index(index))
        if self._reasonseg is not None and rng.random() < self.cfg.data.reasonseg_train_share:
            return self._reasonseg.sample(rng)
        set_name = rng.choices(self.sets, weights=self.weights)[0]
        image_id = rng.choice(self._images[set_name])
        ctx = ImageContext(self.cfg, self.adapter, "train", image_id)
        conversations = []
        for record in self._pick(set_name, self._records[set_name][image_id], rng):
            query = self._query(record, rng)
            conversations.append(record_conversation(ctx, self.spec, record, query, templates.random_answer(rng)))
        return ctx.sample(self.spec, conversations, "labeled")


# ----------------------------------------------------------------------------------------------
# Unlabeled images
# ----------------------------------------------------------------------------------------------
class UnlabeledDataset(Dataset):
    """Unlabeled images with pseudo-labels and K class queries per sampled class (Sec. 3.2.1, Fig. 3).

    Per image, ``classes_per_image`` classes are drawn among those covering at least ``min_class_pixels``
    pixels of the pseudo-label (another image is tried when none does). For each class K queries are drawn
    from the database D: the first trains the model, the other K-1 go to ``aux_queries``. The target is
    ``pseudo_label == class``; the same answer template is used for all K queries of a class. Every listed
    unlabeled image must have a stored pseudo-label, so that an unfinished ``tools/make_pseudo_labels.py`` run
    cannot silently shrink Stage 3; ``allow_partial=True`` (debugging, e.g. a ``--limit`` run) keeps only the
    images that have one and logs a warning. No pseudo-label at all is always an error.
    """

    def __init__(self, cfg: Config, spec: TokenSpec, seed: int = 0, allow_partial: bool = False) -> None:
        self.cfg, self.spec, self.seed = cfg, spec, seed
        ucfg = cfg.unlabeled
        self.adapter = get_adapter(cfg)
        self.pseudo_dir = cfg.pseudo_dir
        if not self.pseudo_dir.is_dir():
            raise FileNotFoundError(f"pseudo-label folder not found: {self.pseudo_dir} (run tools/make_pseudo_labels.py)")
        listed = self.adapter.partition(cfg.data.partition, "unlabeled")
        self.image_ids = [i for i in listed if has_prediction(self.pseudo_dir, i)]
        if not self.image_ids:
            raise FileNotFoundError(f"none of the {len(listed)} unlabeled images of partition {cfg.data.partition} has a pseudo-label in {self.pseudo_dir}")
        if len(self.image_ids) < len(listed):
            message = (f"{len(listed) - len(self.image_ids)} of {len(listed)} unlabeled images of partition "
                       f"{cfg.data.partition} have no pseudo-label in {self.pseudo_dir}")
            if not allow_partial:
                raise FileNotFoundError(f"{message}; finish tools/make_pseudo_labels.py or pass allow_partial=True")
            LOG.warning("%s and are skipped (allow_partial)", message)
        # The confidence map is the weight of the confidence-threshold baseline only; it is neither
        # decoded nor shipped to the device under the default consistency weighting.
        self.use_confidence = ucfg.weighting == "confidence"
        if self.use_confidence:
            without_conf = sum(not prediction_paths(self.pseudo_dir, i)[1].is_file() for i in self.image_ids)
            if without_conf == len(self.image_ids):
                raise FileNotFoundError(f"unlabeled.weighting='confidence' needs the confidence maps, but none of the "
                                        f"{without_conf} pseudo-labels in {self.pseudo_dir} has one (rerun tools/make_pseudo_labels.py)")
            if without_conf:
                LOG.warning("%d of %d pseudo-labels in %s have no confidence file; their confidence is None",
                            without_conf, len(self.image_ids), self.pseudo_dir)
        self.query_db = QueryDB.for_config(cfg)
        if self.query_db.dataset != cfg.data.dataset:
            raise ValueError(f"the query database is for {self.query_db.dataset!r}, the config asks for {cfg.data.dataset!r}")
        short = {c: len(self.query_db.queries(c)) for c in self.query_db.class_ids() if len(self.query_db.queries(c)) < ucfg.num_queries}
        if short:
            raise ValueError(f"classes with fewer than K={ucfg.num_queries} queries in the query database: {short}")
        self.query_db.validate(min_queries=ucfg.num_queries)  # every query names the class it is stored under
        self._classes = frozenset(self.query_db.class_ids())

    def __len__(self) -> int:
        return len(self.image_ids)

    def _eligible(self, label: np.ndarray) -> List[int]:
        counts = np.bincount(label.ravel(), minlength=256)
        return [c for c in sorted(self._classes) if counts[c] >= self.cfg.unlabeled.min_class_pixels]

    def __getitem__(self, index: int) -> Sample:
        index = _check_index(index)
        rng = stream_rng(self.seed, index)
        n = len(self.image_ids)
        first = index % n
        candidates = [first] + [i for i in rng.sample(range(n), min(n, MAX_RESAMPLES)) if i != first]
        for position in candidates[:MAX_RESAMPLES]:
            image_id = self.image_ids[position]
            if self.use_confidence:
                label, confidence = read_prediction(self.pseudo_dir, image_id)
            else:
                label, confidence = _read_label(self.pseudo_dir, image_id), None
            eligible = self._eligible(label)
            if eligible:
                break
        else:
            raise ValueError(f"no image tried has a pseudo-labelled class with at least {self.cfg.unlabeled.min_class_pixels} pixels")
        ctx = ImageContext(self.cfg, self.adapter, "train", image_id)
        if label.shape != ctx.image.shape[:2]:
            raise ValueError(f"{image_id}: pseudo-label shape {label.shape} differs from image shape {ctx.image.shape[:2]}")
        ucfg = self.cfg.unlabeled
        conversations = []
        for class_id in rng.sample(eligible, min(ucfg.classes_per_image, len(eligible))):
            queries = self.query_db.sample(class_id, ucfg.num_queries, rng)
            conversations.append(Conversation(
                query=queries[0], answer=templates.random_answer(rng),
                mask=torch.from_numpy((label == class_id).astype(np.uint8)), class_id=class_id, set_name=C.SET_UNLABELED,
                aux_queries=queries[1:], confidence=None if confidence is None else torch.from_numpy(confidence),
                record_id=f"{image_id}#{class_id}",
            ))
        return ctx.sample(self.spec, conversations, "unlabeled")


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
class EvalDataset(Dataset):
    """One sample per stored instruction of ``sets`` on ``split`` (Sec. 4.1).

    The stored query is used as written (no template redraw), the answer is the fixed teacher-forced
    ``"It is <SSEG>."``, and the target is ``{0, 1}`` (255 on void pixels when ``cfg.data.eval_ignore_void``).
    Each conversation's ``set_name`` is the group of :func:`eval_group`: the LLM-generated conditional records are
    ``conditional``, the rule-based ones ``conditional_templated``.
    """

    def __init__(self, cfg: Config, spec: TokenSpec, sets: Sequence[str], split: str = "val") -> None:
        if split not in ("train", "val"):
            raise ValueError(f"unknown split {split!r}")
        bad = [s for s in sets if s not in C.LABELED_SETS]
        if bad or not sets:
            raise ValueError(f"sets must be a non-empty subset of {C.LABELED_SETS}, got {list(sets)}")
        self.cfg, self.spec, self.split = cfg, spec, split
        self.adapter = get_adapter(cfg)
        self.records: List[InstructionRecord] = []
        for set_name in dict.fromkeys(sets):
            self.records.extend(load_instruction_records(cfg, set_name, split))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.records):
            raise IndexError(f"index {index!r} is outside 0..{len(self.records) - 1}")
        record = self.records[int(index)]
        ctx = ImageContext(self.cfg, self.adapter, self.split, record.image_id)
        conversation = record_conversation(
            ctx, self.spec, record, record.query, EVAL_ANSWER, self.cfg.data.eval_ignore_void, set_name=eval_group(record)
        )
        return ctx.sample(self.spec, [conversation], "eval")
