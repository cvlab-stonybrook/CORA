"""Configuration for CORA.

Every default follows the paper (arXiv 2511.17755 and its supplement). Where the paper is silent or
inconsistent the default is a documented choice, and every alternative stays selectable here. Values are
loaded as: dataclass defaults <- YAML file(s) (``_base_`` chains) <- ``key.subkey=value`` overrides.
"""
from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml

from cora import constants as C

REPO_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Folder that holds data/, outputs/ and pretrained/ (``CORA_ROOT`` env, else the repository root)."""
    return Path(os.environ.get("CORA_ROOT", REPO_ROOT))


# ----------------------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------------------
@dataclass
class PathsConfig:
    """Locations. Relative paths are resolved against :func:`project_root`."""

    data_dir: str = "data"  # derived data: objects, instruction sets, query database, pseudo-labels
    output_dir: str = "outputs"  # checkpoints, logs, evaluation results
    sam_checkpoint: str = "pretrained/sam_vit_h_4b8939.pth"
    cityscapes_root: str = "datasets/cityscapes"
    reasonseg_root: str = "datasets/reasonseg"
    # Optional JSON giving the original long/short flag per ReasonSeg image id (the parquet copies lack it).
    reasonseg_manifest: Optional[str] = None


@dataclass
class DataConfig:
    dataset: str = "cityscapes"  # cityscapes (the only supported dataset)  (Sec. 3.4.1)
    partition: str = "1_30"  # labeled fraction: 1_30 | 1_16 | 1_8 | 1_4  (Sec. 4.2)
    ssl_method: str = "unimatch"  # source of pseudo-labels: unimatch | semivl  (Sec. 4.1)
    # Labeled data: how many instructions (conversations) are drawn per sampled image, as LISA's
    # num_classes_per_sample.
    queries_per_image: int = 3
    # How the CLIP view of the image is made: "pad" pads to a square with the CLIP mean colour and
    # resizes (LLaVA-1.5 behaviour, keeps the whole frame); "center_crop" is the plain CLIP processor
    # (LISA behaviour, drops the sides of a 2:1 Cityscapes frame); "squash" resizes without padding.
    clip_preprocess: str = "pad"
    sam_image_size: int = 1024  # SAM ResizeLongestSide target (LISA)
    max_text_tokens: int = 384  # text tokens kept per conversation (image tokens not counted)
    num_workers: int = 8
    # Objects from semantic masks (supplement Sec. 4): connected components of every class mask.
    connectivity: int = 8
    min_region_area: int = 1  # components smaller than this many pixels get no region id
    # Target masks for evaluation mark void pixels (label 255) as ignored when True.
    eval_ignore_void: bool = False
    # The conditional set of the labeled images has two sources: region-level instructions written by an LLM
    # (the provided instruction files; supplement Table 5, Fig. 3) and class-level instructions built by rule from
    # the masks ("Segment the cars on the left of <REF>", Fig. 2).
    #   "both": each drawn conditional instruction comes from either source with equal probability (default)
    #   "llm" / "templated": one source only
    # An image that has records of one source only always uses that source.
    conditional_sources: str = "both"
    # Share of the labeled samples (draws; a ReasonSeg draw holds one query, a Cityscapes draw up to
    # queries_per_image) that is taken from the ReasonSeg train split instead of the dataset's instruction sets, in
    # every stage. 0 = none (the paper trains on the dataset's own instruction sets only); LISA's own data mix draws
    # 1 sample in 16 from ReasonSeg. Needs tools/prepare_reasonseg.py --split train, which keeps the images with a usable mask
    # (232 of the 239 in the copy used here) and one query per image; LISA's explanation answers are not reproduced.
    reasonseg_train_share: float = 0.0
    # Number of ReasonSeg train images the draws come from (null = all of them). The images are the first ones of one
    # fixed shuffled order of the split (cora.data.reasonseg.train_subset), so a smaller number is a subset of a larger
    # one and the choice does not depend on train.seed: a labeled fraction of ReasonSeg next to the one of the dataset
    # (8, 15, 29, 58 of the 232 images for 1/30, 1/16, 1/8, 1/4, rounded up as the Cityscapes partitions are). The
    # share above stays the share of the draws. Not a setting of the paper.
    reasonseg_train_images: Optional[int] = None


@dataclass
class ReferenceConfig:
    """Anchor (reference object) encoding for conditional instructions (Sec. 3.3; SegLLM encoders)."""

    # "tokens": REF_TOKEN -> anchor mask embedding + anchor box embedding (the paper's method).
    # "text":   REF_TOKEN -> the word "mask" (supplement Sec. 5; format for models without the encoders).
    mode: str = "tokens"
    # Projection of the CLIP feature of the masked anchor crop to the LLM width:
    # "separate" = its own trainable MLP initialised from LLaVA's projector; "shared" = LLaVA's projector
    # itself (SegLLM code; it is frozen in the LISA setup unless train_projector is set).
    mask_projector: str = "separate"
    train_projector: bool = False  # only used with mask_projector="shared"
    # SegLLM's released code drops each reference embedding with probability 0.5 in training and scales it
    # by 0.5 at test time. Neither paper describes this, so it is off by default.
    modality_dropout: float = 0.0
    # Where the two reference positions go: "inline" (where REF_TOKEN stands in the query, as written in
    # the paper's examples) or "prefix" (right after the image, before the query text, as in SegLLM data).
    placement: str = "inline"


@dataclass
class ModelConfig:
    # LLaVA-v1.5-7B with CLIP ViT-L/14-336 (Sec. 4.1): a hub id, or a folder (absolute or relative to the project
    # root) such as the "llava" folder written by tools/convert_lisa.py for a released LISA checkpoint.
    llava: str = "llava-hf/llava-1.5-7b-hf"
    # Starting point taken from a converted LISA checkpoint (tools/convert_lisa.py), for the
    # runs that start from LISA instead of LLaVA + SAM: the file with LISA's trained <SSEG> projection
    # (text_hidden_fcs), and whether the <SSEG> rows of the embedding and of the output head are the checkpoint's
    # (LISA's [SEG] rows, kept) or new rows set to the mean of the vocabulary (the default).
    text_hidden_fcs_init: Optional[str] = None
    pretrained_seg_token: bool = False
    sam_type: str = "vit_h"  # SAM ViT-H vision backbone (Sec. 4.1, following LISA)
    # LISA's llava_v1 conversation template ("curious human"), which LISA uses with LLaVA-1.5; LLaVA-1.5 itself
    # was instruction-tuned with the vicuna_v1 wording ("curious user ... the user's questions").
    system_prompt: str = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    )
    out_dim: int = 256  # width of the <SSEG> projection = SAM prompt width (LISA text_hidden_fcs)
    # LoRA on the language model only (Fig. 3; LISA settings)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    train_mask_decoder: bool = True  # decoder is trainable (Fig. 3)
    # Input embeddings and lm_head are trained in full so that <SSEG> can be learned, as in LISA (Fig. 3 draws the
    # MLLM as LoRA-only); they make up most of a checkpoint (about 1 GB). False gives the literal figure.
    train_embeddings: bool = True
    # The <SSEG> projection (LISA's text_hidden_fcs, 17.8 M values for a 7B model) is trained, as in LISA; Fig. 3 does
    # not draw it. False keeps it frozen, which needs trained starting values (model.text_hidden_fcs_init).
    train_seg_projection: bool = True
    gradient_checkpointing: bool = True
    precision: str = "bf16"  # bf16 | fp32
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    # Tiny randomly initialised LLaVA + SAM built from configs (unit tests and CPU smoke runs only).
    tiny: bool = False


@dataclass
class LossConfig:
    """Labeled-image loss, Eq. 4 and Eq. 5:  L^l = alpha * L_ce + BCE + DICE."""

    ce_weight: float = 1.0  # alpha (Sec. 4.1)
    bce_weight: float = 1.0  # Eq. 4 has unit weights; LISA's code uses 2.0
    dice_weight: float = 1.0  # Eq. 4 has unit weights; LISA's code uses 0.5


@dataclass
class UnlabeledConfig:
    """Unlabeled images (Sec. 3.2.1): pseudo-labels weighted per pixel, Eq. 6 and Eq. 7."""

    weight: float = 0.001  # sigma in Eq. 9 (Sec. 4.1; supplement Table 3)
    batch_size: int = 2  # unlabeled images per device and step
    num_queries: int = 7  # K rephrasings per class: 1 trains the model, K-1 measure consistency (Sec. 4.1)
    classes_per_image: int = 1  # classes sampled from the pseudo-label of an image per step
    min_class_pixels: int = 256  # a class must cover at least this many pseudo-label pixels to be sampled
    # Per-pixel weight lambda_i:
    #   "consistency": from the variance of the K-1 predictions (output consistency, the paper's method);
    #   "confidence":  1 where the semi-supervised model's confidence >= confidence_threshold, else 0
    #                  (the baseline of the ablation and of supplement Table 2);
    #   "none":        1 everywhere.
    weighting: str = "consistency"
    # How the variance v_i becomes lambda_i (the text says stable pixels are emphasised, Eq. 6 prints the
    # variance itself):
    #   "stable_high": lambda = 1 - v / 0.25 clipped to [0, 1]   (default)
    #   "variance":    lambda = v                               (Eq. 6 exactly as printed)
    #   "one_minus":   lambda = 1 - v
    #   "exp":         lambda = exp(-exp_scale * v)
    weight_mode: str = "stable_high"
    exp_scale: float = 10.0
    variance_input: str = "prob"  # variance of sigmoid probabilities ("prob") or of binarised masks ("binary")
    confidence_threshold: float = 0.9  # sigma_sseg >= 0.9 (supplement Sec. 2)
    # Eq. 7 sums over all W*H pixels: "sum" (default) or "mean".
    reduction: str = "sum"
    # DICE cannot be evaluated per pixel, so it enters Eq. 7 as one image-level value D multiplied by the
    # summed (or averaged) weights, sum_i lambda_i * D:
    #   "plain":             D = soft Dice of the two masks (Eq. 7 read literally; default)
    #   "weighted":          D = soft Dice with lambda inside its sums, so unstable pixels also leave the Dice
    #   "weighted_unscaled": the lambda-weighted Dice alone, not multiplied by the weights
    #   "off":               BCE only
    dice_mode: str = "plain"
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    # The K-1 rephrasings go through a network that is not trained by them (Fig. 3):
    #   "online": the current model in eval mode without gradients (default)
    #   "frozen": a snapshot of the weights taken when the unlabeled stage starts
    #   "ema":    an exponential moving average of the trained weights
    teacher: str = "online"
    ema_momentum: float = 0.999
    text_loss: bool = False  # Eq. 9 has no language-modelling term for unlabeled images
    # Train on the unlabeled images that have a pseudo-label even when some are missing (short test runs).
    allow_partial: bool = False


@dataclass
class TFCAConfig:
    """Token-level feature consistency alignment (Sec. 3.2.2, Eq. 8, Fig. 4)."""

    enabled: bool = True
    temperature: float = 0.07  # tau; not given in the paper, GLUS code value
    # Which <SSEG> feature is contrasted: "projected" (the 256-d prompt given to the decoder, as GLUS does)
    # or "hidden" (the LLM last-layer state).
    feature: str = "projected"
    bank_size: int = 128  # labeled-image tokens kept per class (FIFO); not given in the paper
    num_positives: int = 1  # positives sampled per unlabeled token (GLUS: 1); 0 = all same-class tokens
    num_negatives: int = 128  # negatives sampled per unlabeled token (GLUS: 128); 0 = all other-class tokens
    reduction: str = "sum"  # Eq. 8 sums over (v, k+) pairs; "mean" averages over them
    # Fig. 4 also draws a "repel" arrow between labeled tokens of different classes, which Eq. 8 does not
    # contain. When true, the labeled <SSEG> tokens of each step are used as anchors of the same loss as well.
    labeled_anchor: bool = False


@dataclass
class StageConfig:
    """One training stage (Sec. 3.3)."""

    name: str = "stage1"
    labeled_sets: List[str] = field(default_factory=lambda: [C.SET_SEMANTIC, C.SET_ATTRIBUTE])
    set_weights: Optional[List[float]] = None  # sampling weights of labeled_sets (None = uniform)
    use_reference: bool = False  # anchor encoders active (needed for the conditional set)
    use_unlabeled: bool = False  # unlabeled images with pseudo-labels (Eq. 9)
    iters: Optional[int] = None  # None = train.iters_per_stage


def default_stages() -> List[StageConfig]:
    """The three stages of Sec. 3.3, each with the same number of iterations."""
    return [
        # Stage 1: semantic and attribute-based instructions, to learn the object classes.
        StageConfig(name="stage1", labeled_sets=[C.SET_SEMANTIC, C.SET_ATTRIBUTE]),
        # Stage 2: conditional (spatial) visual instructions with the anchor mask and box encoders.
        StageConfig(name="stage2", labeled_sets=[C.SET_CONDITIONAL], use_reference=True),
        # Stage 3: labeled data plus unlabeled images with pseudo-labels (Eq. 9).
        StageConfig(
            name="stage3",
            labeled_sets=[C.SET_SEMANTIC, C.SET_ATTRIBUTE, C.SET_CONDITIONAL],
            use_reference=True,
            use_unlabeled=True,
        ),
    ]


@dataclass
class TrainConfig:
    stages: List[StageConfig] = field(default_factory=default_stages)
    iters_per_stage: int = 5000  # equal for all stages (Sec. 3.3); the count is not in the paper
    batch_size: int = 2  # labeled images per device (Sec. 4.1: batch size 8 = 2 per device on 4 GPUs)
    grad_accum_steps: int = 1
    lr: float = 2e-5  # Sec. 4.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])  # LISA
    weight_decay: float = 0.0  # LISA
    warmup_steps: int = 100  # LISA WarmupDecayLR: linear warm-up, then linear decay to zero over the stage
    schedule: str = "warmup_linear_decay"  # | "constant"
    grad_clip: float = 1.0  # LISA
    seed: int = 0
    log_every: int = 10
    save_every: int = 1000
    eval_every: int = 0  # 0 = evaluate only at the end of a stage
    resume: bool = True  # continue from the newest checkpoint in the run folder
    run_name: str = "cora"


@dataclass
class EvalConfig:
    sets: List[str] = field(default_factory=lambda: [C.SET_CONDITIONAL, C.SET_ATTRIBUTE, C.SET_SEMANTIC])
    # "teacher_forced": the answer containing <SSEG> is given, as in LISA's validation;
    # "generate": the model writes the answer and masks are decoded from the <SSEG> tokens it emits.
    mode: str = "teacher_forced"
    max_new_tokens: int = 32
    batch_size: int = 1
    # Evaluate an evenly spread subset of this many samples (None = all). ``max_samples`` applies to
    # tools/evaluate.py and, when ``stage_max_samples`` is None, to the evaluations during training too;
    # ``stage_max_samples`` limits only the evaluations during training (end of a stage, train.eval_every).
    max_samples: Optional[int] = None
    stage_max_samples: Optional[int] = None


@dataclass
class RuleSetsConfig:
    """Rule-based instruction sets (semantic, templated conditional, offline stand-in; Sec. 3.4.2, supplement Sec. 8)."""

    # Offline stand-in for the attribute set: classes per image, phrases per class.
    attribute_targets_per_image: int = 6
    qa_per_target: int = 3
    # Class-level conditional instructions built by rule from the masks, e.g. "Segment the cars on the left
    # of <REF>" (Fig. 2). Number per image; 0 = off.
    templated_conditional_per_image: int = 4
    seed: int = 0


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    unlabeled: UnlabeledConfig = field(default_factory=UnlabeledConfig)
    tfca: TFCAConfig = field(default_factory=TFCAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    rule_sets: RuleSetsConfig = field(default_factory=RuleSetsConfig)

    # ---- derived locations -------------------------------------------------------------------
    def resolve(self, path: str) -> Path:
        p = Path(os.path.expanduser(path))
        return p if p.is_absolute() else project_root() / p

    @property
    def data_dir(self) -> Path:
        """<data_dir>/<dataset>: objects/, instructions/, query_db.json, pseudo/ live below it."""
        return self.resolve(self.paths.data_dir) / self.data.dataset

    @property
    def run_dir(self) -> Path:
        return self.resolve(self.paths.output_dir) / self.train.run_name

    @property
    def llava_source(self) -> str:
        """What ``from_pretrained`` gets for ``model.llava``: the folder when it exists, else the hub id as written."""
        folder = self.resolve(self.model.llava)
        return str(folder) if folder.is_dir() else self.model.llava

    @property
    def reasonseg_train_dir(self) -> Path:
        """Folder written by ``tools/prepare_reasonseg.py --split train`` (images, masks, index.json)."""
        return self.resolve(self.paths.data_dir) / "reasonseg" / "train"

    def objects_dir(self, split: str) -> Path:
        return self.data_dir / "objects" / split

    def instructions_path(self, split: str, set_name: str, variant: str = "") -> Path:
        """File of one instruction set. ``variant`` names a second source of the same set, e.g.
        ``conditional`` (LLM-generated) and ``conditional_templated`` (rule-based)."""
        name = f"{set_name}_{variant}" if variant else set_name
        return self.data_dir / "instructions" / split / f"{name}.jsonl"

    def instruction_files(self, split: str, set_name: str) -> List[Path]:
        """Every existing file of a set: ``<set>.jsonl`` and its variants ``<set>_*.jsonl``."""
        folder = self.data_dir / "instructions" / split
        files = [folder / f"{set_name}.jsonl"] + sorted(folder.glob(f"{set_name}_*.jsonl"))
        return [f for f in files if f.is_file()]

    @property
    def query_db_path(self) -> Path:
        return self.data_dir / "query_db.json"

    @property
    def pseudo_dir(self) -> Path:
        return self.data_dir / "pseudo" / self.data.ssl_method / self.data.partition

    @property
    def dataset_root(self) -> Path:
        return self.resolve(self.paths.cityscapes_root)


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
_CHOICES = {
    "data.dataset": C.DATASETS,
    # "debug" holds synthetic pseudo-labels made by tools/make_pseudo_labels.py for pipeline tests only.
    "data.ssl_method": ("unimatch", "semivl", "debug"),
    "data.clip_preprocess": ("pad", "center_crop", "squash"),
    "data.connectivity": (4, 8),
    "data.conditional_sources": ("both", "llm", "templated"),
    "model.precision": ("bf16", "fp32"),
    "model.reference.mode": ("tokens", "text"),
    "model.reference.mask_projector": ("separate", "shared"),
    "model.reference.placement": ("inline", "prefix"),
    "unlabeled.weighting": ("consistency", "confidence", "none"),
    "unlabeled.weight_mode": ("stable_high", "variance", "one_minus", "exp"),
    "unlabeled.variance_input": ("prob", "binary"),
    "unlabeled.reduction": ("sum", "mean"),
    "unlabeled.dice_mode": ("plain", "weighted", "weighted_unscaled", "off"),
    "unlabeled.teacher": ("online", "frozen", "ema"),
    "tfca.feature": ("projected", "hidden"),
    "tfca.reduction": ("sum", "mean"),
    "train.schedule": ("warmup_linear_decay", "constant"),
    "eval.mode": ("teacher_forced", "generate"),
}


def _coerce(where: str, value: Any, default: Any) -> Any:
    """Convert a YAML value to the type of the option's default (PyYAML reads ``2e-5`` as a string)."""
    if value is None or default is None:
        return value
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
            raise ValueError("expected true or false")
        if isinstance(default, float):
            if isinstance(value, bool):
                raise ValueError("expected a number")
            return float(value)
        if isinstance(default, int):
            if isinstance(value, bool):
                raise ValueError("expected an integer")
            if isinstance(value, float):
                if not value.is_integer():
                    raise ValueError("expected an integer")
                return int(value)
            return int(value)
        if isinstance(default, list) and isinstance(value, list) and default and isinstance(default[0], float):
            return [float(v) for v in value]
        if isinstance(default, str) and not isinstance(value, str):
            raise ValueError('expected text; quote it in YAML (for example "1_30", which YAML reads as 130)')
    except (TypeError, ValueError) as err:
        raise ValueError(f"{where}={value!r}: {err}") from None
    return value


def _from_dict(cls, data: Any):
    """Build dataclass ``cls`` from a (possibly partial) nested dict, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    defaults = cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise KeyError(f"unknown option(s) for {cls.__name__}: {unknown}; valid: {sorted(known)}")
    kwargs = {}
    for name, value in data.items():
        if cls is TrainConfig and name == "stages":
            kwargs[name] = [_from_dict(StageConfig, s) for s in value]
            continue
        sub = _SECTION_TYPES.get((cls.__name__, name))
        if sub is not None:
            kwargs[name] = _from_dict(sub, value)
        else:
            kwargs[name] = _coerce(f"{cls.__name__}.{name}", value, getattr(defaults, name))
    return cls(**kwargs)


_SECTION_TYPES = {
    ("Config", "paths"): PathsConfig,
    ("Config", "data"): DataConfig,
    ("Config", "model"): ModelConfig,
    ("Config", "loss"): LossConfig,
    ("Config", "unlabeled"): UnlabeledConfig,
    ("Config", "tfca"): TFCAConfig,
    ("Config", "train"): TrainConfig,
    ("Config", "eval"): EvalConfig,
    ("Config", "rule_sets"): RuleSetsConfig,
    ("ModelConfig", "reference"): ReferenceConfig,
}


def _merge(base: dict, update: dict) -> dict:
    """Recursively merge ``update`` into ``base`` (lists and scalars are replaced)."""
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path, _seen=()) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular _base_ chain at {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    bases = data.pop("_base_", None)
    merged: dict = {}
    for base in [bases] if isinstance(bases, str) else (bases or []):
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _merge(merged, _read_yaml(base_path, _seen + (path,)))
    return _merge(merged, data)


def _set_override(data: dict, dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        if key.isdigit() and isinstance(node, list):
            node = node[int(key)]
        else:
            node = node.setdefault(key, {})
    last = keys[-1]
    is_index = last.isdigit() and isinstance(node, list)
    current = node[int(last)] if is_index else (node.get(last) if isinstance(node, dict) else None)
    # String options keep the raw text: YAML would read "1_30" as the integer 130.
    value = raw.strip().strip("\"'") if isinstance(current, str) else yaml.safe_load(raw)
    if is_index:
        node[int(last)] = value
    else:
        node[last] = value


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def validate(cfg: Config) -> None:
    flat = to_dict(cfg)
    for dotted, choices in _CHOICES.items():
        node: Any = flat
        for key in dotted.split("."):
            node = node[key]
        if node not in choices:
            raise ValueError(f"{dotted}={node!r} is not one of {list(choices)}")
    partitions = C.CITYSCAPES_PARTITIONS
    if cfg.data.partition not in partitions:
        raise ValueError(f"data.partition={cfg.data.partition!r} is not one of {list(partitions)}")
    if cfg.unlabeled.num_queries < 2:
        raise ValueError("unlabeled.num_queries (K) must be at least 2: one query trains, K-1 measure consistency")
    if not 0.0 <= cfg.data.reasonseg_train_share < 1.0:
        raise ValueError(f"data.reasonseg_train_share must be in [0, 1), got {cfg.data.reasonseg_train_share}")
    if cfg.data.reasonseg_train_images is not None:
        if cfg.data.reasonseg_train_images < 1:
            raise ValueError(f"data.reasonseg_train_images must be at least 1 or null (all images), got {cfg.data.reasonseg_train_images}")
        if cfg.data.reasonseg_train_share <= 0:
            raise ValueError("data.reasonseg_train_images is set but data.reasonseg_train_share is 0: no ReasonSeg sample would be drawn")
    if not cfg.model.train_seg_projection and not cfg.model.text_hidden_fcs_init and not cfg.model.tiny:
        raise ValueError("model.train_seg_projection=false needs model.text_hidden_fcs_init: a frozen <SSEG> projection "
                         "with random values cannot produce masks")
    if cfg.model.pretrained_seg_token and cfg.model.tiny:
        raise ValueError("model.pretrained_seg_token needs a converted checkpoint (model.llava), not the tiny random model")
    for name in ("max_samples", "stage_max_samples"):
        value = getattr(cfg.eval, name)
        if value is not None and value < 1:
            raise ValueError(f"eval.{name} must be at least 1 or null (all samples), got {value}")
    if not cfg.train.stages:
        raise ValueError("train.stages is empty")
    for stage in cfg.train.stages:
        bad = [s for s in stage.labeled_sets if s not in C.LABELED_SETS]
        if bad:
            raise ValueError(f"stage {stage.name}: unknown labeled set(s) {bad}; valid: {list(C.LABELED_SETS)}")
        if stage.set_weights is not None and len(stage.set_weights) != len(stage.labeled_sets):
            raise ValueError(f"stage {stage.name}: set_weights must match labeled_sets in length")
        if (
            C.SET_CONDITIONAL in stage.labeled_sets
            and cfg.model.reference.mode == "tokens"
            and not stage.use_reference
        ):
            raise ValueError(
                f"stage {stage.name} trains on the conditional set, so use_reference must be true "
                "(or set model.reference.mode=text)"
            )


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Load a configuration.

    ``path`` is a YAML file (may list ``_base_`` files, relative to itself); ``overrides`` are strings
    such as ``"train.lr=1e-5"`` or ``"train.stages.2.iters=100"`` whose values are parsed as YAML.
    """
    data = to_dict(Config())
    if path:
        data = _merge(data, _read_yaml(Path(path)))
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form key=value")
        _set_override(data, key.strip(), raw)
    cfg = _from_dict(Config, data)
    validate(cfg)
    return cfg


def save_config(cfg: Config, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
