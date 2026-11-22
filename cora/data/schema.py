"""On-disk record formats shared by the instruction generators and the datasets.

Instruction sets are JSON-lines files, one :class:`InstructionRecord` per line, stored at
``<data_dir>/<dataset>/instructions/<split>/<set>.jsonl`` (see ``Config.instructions_path``).
Masks are never stored in these files: a record names its target (and anchor) by region id, and the
dataset rebuilds the mask from the image's region map (``cora.data.objects``).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from cora import constants as C


@dataclass
class ObjectRef:
    """A target or an anchor: one or more regions of one class, or a whole class of an image."""

    label_id: int  # class id (Cityscapes trainId 0..18)
    label_name: str
    # Connected-component ids such as "13_1" (supplement Fig. 3). Empty list = every pixel of the class
    # (used by the semantic set and by class-level queries).
    region_ids: List[str] = field(default_factory=list)

    @property
    def is_class_level(self) -> bool:
        return not self.region_ids


@dataclass
class InstructionRecord:
    """One training / evaluation instruction."""

    id: str  # unique within its file
    dataset: str  # cityscapes
    split: str  # train | val
    image_id: str  # dataset-specific image id (see cora.data.adapters)
    set: str  # semantic | attribute | conditional
    # Query text without the image token. Conditional queries contain exactly one REF_TOKEN ("<REF>")
    # where the anchor is referred to.
    query: str
    target: ObjectRef
    anchor: Optional[ObjectRef] = None  # required for the conditional set
    relation: Optional[str] = None  # spatial relation phrase, when known (e.g. "to the right of")
    description: Optional[str] = None  # visual description of the target the query was built from
    # Provenance: generator name, request key, index within the reply, raw Q/A text, ...
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.set not in C.LABELED_SETS:
            raise ValueError(f"{self.id}: unknown set {self.set!r}")
        n_ref = self.query.count(C.REF_TOKEN)
        if self.set == C.SET_CONDITIONAL:
            if self.anchor is None or not self.anchor.region_ids:
                raise ValueError(f"{self.id}: a conditional instruction needs an anchor with region ids")
            if n_ref != 1:
                raise ValueError(f"{self.id}: a conditional query must contain {C.REF_TOKEN} exactly once")
            if set(self.anchor.region_ids) & set(self.target.region_ids):
                raise ValueError(f"{self.id}: anchor and target must be different regions")
        elif n_ref:
            raise ValueError(f"{self.id}: only conditional queries may contain {C.REF_TOKEN}")
        if not self.query.strip():
            raise ValueError(f"{self.id}: empty query")
        for token in (C.IMAGE_TOKEN, C.SEG_TOKEN, C.REF_MASK_TOKEN, C.REF_BOX_TOKEN):
            if token in self.query:
                raise ValueError(f"{self.id}: query must not contain {token}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructionRecord":
        data = dict(data)
        data["target"] = ObjectRef(**data["target"])
        if data.get("anchor") is not None:
            data["anchor"] = ObjectRef(**data["anchor"])
        return cls(**data)


def write_jsonl(path, records: Iterable[InstructionRecord], append: bool = False) -> int:
    """Write records (validated) to ``path``; returns how many were written.

    A new file is written next to ``path`` and renamed over it, so a reader (for example a training run that loads
    the validation sets at the end of a stage) sees either the old file or the complete new one, never a part.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    if append:
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                record.validate()
                f.write(record.to_json() + "\n")
                count += 1
        return count
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for record in records:
                record.validate()
                f.write(record.to_json() + "\n")
                count += 1
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return count


def iter_jsonl(path) -> Iterator[InstructionRecord]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield InstructionRecord.from_dict(json.loads(line))
            except Exception as err:  # corrupt line: say where
                raise ValueError(f"{path}:{line_no}: {err}") from err


def read_jsonl(path) -> List[InstructionRecord]:
    return list(iter_jsonl(path))
