"""Shared constants: special tokens, label conventions and class tables."""

# ----------------------------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------------------------
IGNORE_INDEX = -100  # token label ignored by the language-modelling loss
IGNORE_LABEL = 255  # "void" value in semantic label maps, pseudo-label maps and evaluation targets

# ----------------------------------------------------------------------------------------------
# Special tokens
# ----------------------------------------------------------------------------------------------
IMAGE_TOKEN = "<image>"  # LLaVA image placeholder (one per prompt; expanded to image_seq_len positions)
SEG_TOKEN = "<SSEG>"  # segmentation token of the paper (Sec. 3.1); plays the role of LISA's [SEG]
# Stored queries mark the anchor (reference object) with REF_TOKEN, exactly like "<REF>" in Fig. 2 / Fig. 6.
REF_TOKEN = "<REF>"
# At prompt-building time REF_TOKEN becomes two LLM positions, the SegLLM "<mask> <box>" pair
# (supplement Sec. 5): one anchor-mask embedding followed by one anchor-box embedding.
REF_MASK_TOKEN = "<REFMASK>"
REF_BOX_TOKEN = "<REFBOX>"
# Text-only format for models without reference encoders (supplement Sec. 5):
# "Segment the car to the right of <mask> <box>." -> "Segment the car to the right of the mask."
REF_TEXT_FALLBACK = "the mask"
# Tokens that LISA's released checkpoints (built on LLaVA-Lightning-7B-v1-1, mm_use_im_start_end) write around
# the image features. A tokenizer that has them makes every prompt use LISA's layout (cora.data.conversation).
IM_START_TOKEN = "<im_start>"
IM_END_TOKEN = "<im_end>"

# Instruction-set names (Sec. 3.4.2)
SET_SEMANTIC = "semantic"
SET_ATTRIBUTE = "attribute"
SET_CONDITIONAL = "conditional"
SET_UNLABELED = "unlabeled"  # class queries drawn from the per-class query database D (Sec. 3.2.1)
LABELED_SETS = (SET_SEMANTIC, SET_ATTRIBUTE, SET_CONDITIONAL)

# ----------------------------------------------------------------------------------------------
# Cityscapes (19 evaluation classes, trainId order; official cityscapesScripts labels.py)
# ----------------------------------------------------------------------------------------------
CITYSCAPES_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
)
CITYSCAPES_LABELID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
    26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}
CITYSCAPES_PALETTE = (
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
    (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
)
# Region ids starting with "11_" (person) or "12_" (rider) are described as "individuals"
# (supplement Tables 4 and 5).
CITYSCAPES_PERSON_LIKE = (11, 12)
CITYSCAPES_PARTITIONS = ("1_30", "1_16", "1_8", "1_4")

DATASETS = ("cityscapes",)


def class_names(dataset: str) -> tuple:
    """Class names of a dataset, indexed by class id."""
    if dataset == "cityscapes":
        return CITYSCAPES_CLASSES
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def region_id(label_id: int, index: int) -> str:
    """Region id "<label id>_<1-based index of the connected component within its class>" (supplement Fig. 3)."""
    return f"{int(label_id)}_{int(index)}"


def parse_region_id(rid: str) -> tuple:
    """Inverse of :func:`region_id`; returns (label_id, index)."""
    label, _, index = rid.partition("_")
    return int(label), int(index)
