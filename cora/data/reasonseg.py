"""ReasonSeg evaluation data (paper Table 3, Sec. 4.2; LISA's ``ValDataset`` protocol).

Two on-disk layouts are read:

* LISA's original folder ``<root>/ReasonSeg/<split>/`` (or ``<root>/<split>/``) holding ``<name>.jpg`` and
  ``<name>.json`` per image. The mask is rasterised from the polygons exactly as LISA's ``get_mask_from_json``
  (1 = target, 255 = ignore region, smaller shapes drawn last), the first text of the JSON is the query and
  ``is_sentence`` the long/short flag.
* The parquet copies ``<root>/reasonseg_<split>/data/*.parquet`` (one query per image, a boolean target mask,
  no ignore regions, no long/short flag). Rows are read lazily, one at a time.

The long/short flag of the parquet copies is joined by ``image_id`` from the JSON file named in
``cfg.paths.reasonseg_manifest`` (``{"<split>": {"images": [{"image_id", "is_sentence"}, ...]}}``, or the same
list under a top-level ``"images"``); it is never guessed from the text. Without it ``is_long_query`` is None
and the long prompt (the text followed by the instruction) is used.
"""
from __future__ import annotations

import io
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from cora import constants as C
from cora.config import Config
from cora.data import templates
from cora.data.batch import Conversation, Sample
from cora.data.transforms import clip_preprocess, sam_preprocess
from cora.tokens import TokenSpec

LOG = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
REASONSEG_ANSWER = f"{C.SEG_TOKEN}."  # LISA validates with the answer "[SEG]."
# LISA's ValDataset writes "<image>\n {text} ..." with a space after the newline; build_prompt adds "<image>\n", so
# both prompts start with that space (it changes the first text token, e.g. "▁What" instead of "What").
LONG_PROMPT = " {text} Please output segmentation mask."
SHORT_PROMPT = " What is {text} in this image? Please output segmentation mask."
PARQUET_BUFFER = 4 << 20  # bytes buffered while streaming a row group


def reasonseg_query(text: str, is_long: Optional[bool]) -> str:
    """LISA's validation prompt: long queries are used as written, short ones are wrapped in a question.

    An unknown flag (``None``) uses the long form, which adds nothing to the text. As in LISA the query starts
    with one space (after the newline that follows the image token); LISA's training prompts have none.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty ReasonSeg text")
    return SHORT_PROMPT.format(text=text) if is_long is False else LONG_PROMPT.format(text=text)


def rasterise_shapes(shapes: List[dict], height: int, width: int) -> np.ndarray:
    """(H, W) uint8 mask {0, 1, 255} from LISA polygon annotations (``get_mask_from_json``).

    Shapes labelled "flag" are skipped; polygons are painted from the largest to the smallest area with 255
    when the lower-cased label contains "ignore" and 1 otherwise.
    """
    polygons: List[Tuple[int, str, np.ndarray]] = []
    for shape in shapes:
        label = str(shape["label"])
        if label.lower() == "flag":
            continue
        points = np.array([shape["points"]], dtype=np.int32)
        scratch = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(scratch, points, True, 1, 1)
        cv2.fillPoly(scratch, points, 1)
        polygons.append((int(scratch.sum()), label, points))
    polygons.sort(key=lambda item: item[0], reverse=True)
    mask = np.zeros((height, width), dtype=np.uint8)
    for _, label, points in polygons:
        value = 255 if "ignore" in label.lower() else 1
        cv2.polylines(mask, points, True, value, 1)
        cv2.fillPoly(mask, points, value)
    return mask


def load_flag_manifest(path: str, split: str) -> Dict[str, bool]:
    """``image_id -> is_sentence`` of one split from a manifest JSON (see the module docstring)."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    section = data.get(split, data) if isinstance(data, dict) else data
    images = section.get("images") if isinstance(section, dict) else section
    if not isinstance(images, list):
        raise ValueError(f"{path}: expected a list of images under '{split}' -> 'images' or 'images'")
    try:
        return {str(item["image_id"]): bool(item["is_sentence"]) for item in images}
    except (KeyError, TypeError) as err:
        raise ValueError(f"{path}: every image entry needs 'image_id' and 'is_sentence' ({err})") from None


class _ParquetRows:
    """Sequential-friendly lazy access to the rows of several parquet files (never a whole file in memory)."""

    COLUMNS = ("image", "mask")

    def __init__(self, files: List[Path]) -> None:
        self.files = files
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.locations: List[Tuple[int, int, int]] = []  # file, row group, offset in the group
        for f, path in enumerate(files):
            parquet = pq.ParquetFile(path)
            table = pq.read_table(path, columns=["image_id", "text"])
            self.ids.extend(table.column("image_id").to_pylist())
            self.texts.extend(table.column("text").to_pylist())
            for g in range(parquet.metadata.num_row_groups):
                self.locations.extend((f, g, k) for k in range(parquet.metadata.row_group(g).num_rows))
        if len(self.locations) != len(self.ids):
            raise ValueError("parquet row count differs from the row-group metadata")
        self._cursor: Optional[Tuple[int, int, int, object]] = None  # file, group, next offset, batch iterator

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_cursor"] = None  # iterators are not picklable; workers open their own
        return state

    def read(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """(RGB image, bool mask) of a row; consecutive rows of a group are streamed, not re-read."""
        f, g, k = self.locations[index]
        cursor = self._cursor
        if cursor is None or cursor[:2] != (f, g) or cursor[2] > k:
            parquet = pq.ParquetFile(self.files[f], buffer_size=PARQUET_BUFFER)
            iterator = parquet.iter_batches(batch_size=1, row_groups=[g], columns=list(self.COLUMNS))
            cursor = (f, g, 0, iterator)
        _, _, position, iterator = cursor
        batch = None
        while position <= k:
            batch = next(iterator)
            position += 1
        self._cursor = (f, g, position, iterator)
        # The stored JPEG bytes keep their EXIF orientation tag and the masks are drawn in the rotated frame (LISA
        # reads the images with cv2.imread, which applies the tag), so the tag is applied here as well.
        picture = ImageOps.exif_transpose(Image.open(io.BytesIO(batch.column("image")[0]["bytes"].as_py())))
        image = np.asarray(picture.convert("RGB"))
        rows = batch.column("mask").flatten()
        if len(rows) == 0:
            raise ValueError(f"{self.ids[index]}: empty mask array")
        flat = rows.flatten().to_numpy(zero_copy_only=False)
        return image, flat.reshape(len(rows), -1).astype(bool)


class ReasonSegDataset(Dataset):
    """ReasonSeg images with LISA's validation prompt; one :class:`Sample` (kind ``"eval"``) per image."""

    def __init__(self, cfg: Config, spec: TokenSpec, split: str = "test") -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown ReasonSeg split {split!r}; expected one of {SPLITS}")
        self.cfg, self.spec, self.split = cfg, spec, split
        root = cfg.resolve(cfg.paths.reasonseg_root)
        self._folder: Optional[Path] = None
        self._rows: Optional[_ParquetRows] = None
        for candidate in (root / "ReasonSeg" / split, root / split):
            if candidate.is_dir() and any(candidate.glob("*.jpg")):
                self._folder = candidate
                break
        if self._folder is not None:
            self._names = sorted(p.stem for p in self._folder.glob("*.jpg"))
            missing = [n for n in self._names if not (self._folder / f"{n}.json").is_file()]
            if missing:
                raise FileNotFoundError(f"{self._folder}: no .json file for {len(missing)} images, first: {missing[0]}")
            self.image_ids = list(self._names)
        else:
            files = sorted((root / f"reasonseg_{split}" / "data").glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no ReasonSeg {split} data under {root}: expected {root / 'ReasonSeg' / split}/*.jpg or {root / f'reasonseg_{split}' / 'data'}/*.parquet")
            self._rows = _ParquetRows(files)
            self.image_ids = list(self._rows.ids)
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("duplicate ReasonSeg image ids")
        self._flags: Dict[str, bool] = {}
        if cfg.paths.reasonseg_manifest:
            self._flags = load_flag_manifest(str(cfg.resolve(cfg.paths.reasonseg_manifest)), split)
            unknown = [i for i in self.image_ids if i not in self._flags]
            if unknown:
                raise ValueError(f"the manifest has no long/short flag for {len(unknown)} images, first: {unknown[0]}")
        elif self._folder is None:
            LOG.warning("no paths.reasonseg_manifest: the long/short flag of the parquet copies is unknown "
                        "and the long prompt is used for every query")

    def __len__(self) -> int:
        return len(self.image_ids)

    def is_long_query(self, index: int) -> Optional[bool]:
        """Long/short flag of an image: from the folder's JSON, else from the manifest, else None."""
        image_id = self.image_ids[index]
        if self._folder is not None:
            return bool(self._read_json(image_id)["is_sentence"])
        return self._flags.get(image_id)

    def _read_json(self, name: str) -> dict:
        path = self._folder / f"{name}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return json.loads(path.read_text(encoding="cp1252"))

    def _load(self, index: int) -> Tuple[np.ndarray, np.ndarray, str, Optional[bool]]:
        """(image, mask uint8 {0, 1, 255}, text, long flag) of an image."""
        if self._folder is not None:
            name = self.image_ids[index]
            info = self._read_json(name)
            bgr = cv2.imread(str(self._folder / f"{name}.jpg"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"cannot read {self._folder / f'{name}.jpg'}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            texts = info["text"]
            if not texts:
                raise ValueError(f"{name}: the JSON has no text")
            mask = rasterise_shapes(info["shapes"], image.shape[0], image.shape[1])
            return image, mask, str(texts[0]), bool(info["is_sentence"])  # LISA evaluates the first text only
        image, target = self._rows.read(index)
        if target.shape != image.shape[:2]:
            raise ValueError(f"{self.image_ids[index]}: mask shape {target.shape} differs from image shape {image.shape[:2]}")
        return image, target.astype(np.uint8), self._rows.texts[index], self._flags.get(self.image_ids[index])

    def __getitem__(self, index: int) -> Sample:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(f"index {index!r} is outside 0..{len(self) - 1}")
        index = int(index)
        image, mask, text, is_long = self._load(index)
        sam, resize_hw = sam_preprocess(image, self.cfg.data.sam_image_size)
        clip = clip_preprocess(image, self.spec.clip_image_size, self.cfg.data.clip_preprocess)
        conversation = Conversation(
            query=reasonseg_query(text, is_long), answer=REASONSEG_ANSWER, mask=torch.from_numpy(mask), class_id=-1,
            set_name="reasonseg", record_id=self.image_ids[index], is_long_query=is_long,
        )
        return Sample(self.image_ids[index], sam, clip, resize_hw, (image.shape[0], image.shape[1]), [conversation], "eval")


TRAIN_ORDER_SEED = "cora-reasonseg-train-order"


def train_subset(image_ids: Sequence[str], count: int) -> List[str]:
    """The ``count`` images of the ReasonSeg train split that a labeled fraction uses (``data.reasonseg_train_images``).

    They are the first ones of one fixed shuffled order of the sorted ids: every smaller subset is contained in
    every larger one, and the choice depends on nothing but the list of prepared images (not on ``train.seed``).
    """
    ordered = sorted(image_ids)
    if not 1 <= count <= len(ordered):
        raise ValueError(f"data.reasonseg_train_images={count}: the prepared ReasonSeg train split has {len(ordered)} images")
    random.Random(TRAIN_ORDER_SEED).shuffle(ordered)
    return ordered[:count]


class ReasonSegTrainSet:
    """The ReasonSeg train split as labeled training samples (``data.reasonseg_train_share``).

    Reads the folder written by ``tools/prepare_reasonseg.py --split train``. A draw is one image with its
    query in LISA's training wording (``reason_seg_dataset.py``: one of the long templates around a sentence, one
    of the short templates around a lower-cased phrase) and one of LISA's answers. ``class_id`` is -1, so the
    token bank of the contrastive loss ignores these samples. Images whose long-query flag is unknown use the
    long template. Not reproduced from LISA: its several phrasings per image (the parquet copy holds one) and
    its explanation answers (``--explanatory``), for which no data is at hand.
    """

    SET_NAME = "reasonseg"

    def __init__(self, cfg: Config, spec: TokenSpec) -> None:
        self.cfg, self.spec = cfg, spec
        self.folder = cfg.reasonseg_train_dir
        index = self.folder / "index.json"
        if not index.is_file():
            raise FileNotFoundError(
                f"data.reasonseg_train_share={cfg.data.reasonseg_train_share} needs {index}; "
                f"write it with: python tools/prepare_reasonseg.py --split train"
            )
        self.entries: List[dict] = json.loads(index.read_text(encoding="utf-8"))
        if not self.entries:
            raise ValueError(f"{index} lists no image")
        missing = [e["image_id"] for e in self.entries
                   if not (self.folder / f"{e['image_id']}.jpg").is_file() or not (self.folder / f"{e['image_id']}.png").is_file()]
        if missing:
            raise FileNotFoundError(f"{self.folder}: image or mask file missing for {len(missing)} entries, first: {missing[0]}")
        self.available = len(self.entries)  # images of the prepared split; the draws may use a subset of them
        if cfg.data.reasonseg_train_images is not None:
            keep = set(train_subset([e["image_id"] for e in self.entries], cfg.data.reasonseg_train_images))
            self.entries = [e for e in self.entries if e["image_id"] in keep]

    def __len__(self) -> int:
        return len(self.entries)

    def sample(self, rng) -> Sample:
        entry = self.entries[rng.randrange(len(self.entries))]
        image_id = entry["image_id"]
        bgr = cv2.imread(str(self.folder / f"{image_id}.jpg"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.folder / f"{image_id}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask is None:
            raise FileNotFoundError(f"cannot read {image_id} from {self.folder}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"{image_id}: mask shape {mask.shape} differs from image shape {image.shape[:2]}")
        is_long = entry.get("is_sentence")
        text = str(entry["text"]).strip()
        query = templates.semantic_query(text, rng) if is_long is False else templates.long_query(text, rng)
        sam, resize_hw = sam_preprocess(image, self.cfg.data.sam_image_size)
        clip = clip_preprocess(image, self.spec.clip_image_size, self.cfg.data.clip_preprocess)
        conversation = Conversation(
            query=query, answer=templates.random_answer(rng), mask=torch.from_numpy(mask.astype(np.uint8)), class_id=-1,
            set_name=self.SET_NAME, record_id=image_id, is_long_query=is_long,
        )
        return Sample(f"reasonseg/{image_id}", sam, clip, resize_hw, (image.shape[0], image.shape[1]), [conversation], "labeled")
