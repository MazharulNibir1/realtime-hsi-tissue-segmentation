"""Deterministic subset selection for HeiPorSPECTRAL.

Each recording has `annotations/<ts>_meta.json` with
`{"label_meta": {<organ>: {"situs": s, "angle": a, "repetition": r}}}`.
For every (pig, organ) pair we keep one image per situs, preferring the configured
angle/repetition and falling back to the lexicographically smallest (angle, repetition).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ImageRecord:
    image_name: str  # "P086#2021_04_15_09_22_02"
    subject: str
    timestamp: str
    label: str
    situs: int
    angle: int
    repetition: int

    def to_dict(self) -> dict:
        return asdict(self)


def parse_label_meta(image_name: str, meta: dict) -> ImageRecord | None:
    """Return a record for single-organ images; None for images with != 1 organ."""
    label_meta = meta.get("label_meta", {})
    if len(label_meta) != 1:
        return None
    ((label, m),) = label_meta.items()
    subject, timestamp = image_name.split("#")
    return ImageRecord(
        image_name,
        subject,
        timestamp,
        label,
        int(m["situs"]),
        int(m["angle"]),
        int(m["repetition"]),
    )


def select_images(
    records: list[ImageRecord], angle: int = 0, repetition: int = 1, max_per_pair: int | None = None
) -> list[ImageRecord]:
    """One image per (subject, label, situs); returns records sorted by (subject, label, situs)."""
    groups: dict[tuple[str, str, int], list[ImageRecord]] = defaultdict(list)
    for r in records:
        groups[(r.subject, r.label, r.situs)].append(r)

    def rank(r: ImageRecord):
        return (r.angle != angle or r.repetition != repetition, r.angle, r.repetition, r.timestamp)

    chosen = [min(g, key=rank) for _, g in sorted(groups.items())]
    if max_per_pair is not None:
        per_pair: dict[tuple[str, str], int] = defaultdict(int)
        kept = []
        for r in chosen:
            if per_pair[(r.subject, r.label)] < max_per_pair:
                per_pair[(r.subject, r.label)] += 1
                kept.append(r)
        chosen = kept
    return chosen
