"""Statistics of an instruction set (supplement Sec. 7.1, Fig. 2a and 2b).

Fig. 2a counts the instructions per target class of the conditional training set (bars, sorted by count in
descending order); Fig. 2b gives the average instruction length per class (one marker per class, same order).
The paper does not say whether the length is counted in words or characters, so both are reported.

Two texts are measured per record: ``record.query`` (what the model sees; the anchor is the ``<REF>`` token)
and the paper-style text, which is ``meta["question"]`` when the generator stored it (the LLM sets keep the
original question, in which the anchor is written "region 11_2") and ``record.query`` otherwise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from cora.data.schema import InstructionRecord


@dataclass(frozen=True)
class ClassStats:
    """Instruction count and mean lengths of one target class."""

    label_name: str
    count: int
    mean_words: float
    mean_chars: float
    mean_words_paper: float
    mean_chars_paper: float


def paper_text(record: InstructionRecord) -> str:
    """The question as the paper measures it: ``meta["question"]`` when present, else the query."""
    question = record.meta.get("question")
    return question if isinstance(question, str) and question.strip() else record.query


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values)


def instruction_stats(records: Iterable[InstructionRecord]) -> List[ClassStats]:
    """Per-class statistics, sorted by count (descending), ties by class name (Fig. 2a order)."""
    grouped: Dict[str, List[InstructionRecord]] = {}
    for record in records:
        grouped.setdefault(record.target.label_name, []).append(record)
    if not grouped:
        raise ValueError("no instruction records to summarise")
    rows = []
    for name, items in grouped.items():
        queries = [r.query for r in items]
        papers = [paper_text(r) for r in items]
        rows.append(ClassStats(
            name, len(items),
            _mean([len(q.split()) for q in queries]), _mean([len(q) for q in queries]),
            _mean([len(p.split()) for p in papers]), _mean([len(p) for p in papers]),
        ))
    return sorted(rows, key=lambda row: (-row.count, row.label_name))


def stats_dict(stats: Sequence[ClassStats]) -> dict:
    """JSON-serialisable summary: the per-class rows and the totals."""
    total = sum(s.count for s in stats)
    return {
        "total": total,
        "mean_words": sum(s.mean_words * s.count for s in stats) / total,
        "mean_words_paper": sum(s.mean_words_paper * s.count for s in stats) / total,
        "classes": [asdict(s) for s in stats],
    }


def stats_table(stats: Sequence[ClassStats]) -> str:
    """Plain-text table of the statistics."""
    head = f"{'class':<16}{'count':>7}{'words':>9}{'chars':>9}{'words*':>9}{'chars*':>9}"
    lines = [head, "-" * len(head)]
    lines += [f"{s.label_name:<16}{s.count:>7}{s.mean_words:>9.1f}{s.mean_chars:>9.1f}"
              f"{s.mean_words_paper:>9.1f}{s.mean_chars_paper:>9.1f}" for s in stats]
    total = stats_dict(stats)
    lines.append(f"{'all':<16}{total['total']:>7}{total['mean_words']:>9.1f}{'':>9}{total['mean_words_paper']:>9.1f}")
    lines.append("* paper-style text: meta['question'] when stored, else the query")
    return "\n".join(lines)


def plot_stats(stats: Sequence[ClassStats], path: Path, *, unit: str = "words") -> Path:
    """Draw Fig. 2a (count per class) and Fig. 2b (mean length per class) side by side into a PNG.

    ``unit`` is "words" or "chars"; the paper-style text is measured.
    """
    if unit not in ("words", "chars"):
        raise ValueError(f"unit must be 'words' or 'chars', got {unit!r}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s.label_name for s in stats]
    lengths = [s.mean_words_paper if unit == "words" else s.mean_chars_paper for s in stats]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    left.bar(names, [s.count for s in stats], color="#ff5050")
    left.set_ylabel("Number of instructions")
    right.plot(names, lengths, marker="o", markersize=3, linewidth=0.8, color="#f0a000")
    right.set_ylabel(f"Average length of instructions ({unit})")
    for axis, title in ((left, "(a)"), (right, "(b)")):
        axis.set_xlabel("Class")
        axis.set_title(title, y=-0.55)
        axis.tick_params(axis="x", labelrotation=45)
        for label in axis.get_xticklabels():
            label.set_ha("right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
