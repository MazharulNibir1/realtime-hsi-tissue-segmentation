"""Subject-level (pig-level) split. No pig ever appears in more than one split."""

from __future__ import annotations

from collections.abc import Iterable


def make_split(subjects: Iterable[str], val: list[str], test: list[str]) -> dict[str, list[str]]:
    """train = all subjects not in val/test. Raises on overlap or unknown subjects."""
    subjects = sorted(set(subjects))
    val_s, test_s = set(val), set(test)
    if val_s & test_s:
        raise ValueError(f"Subjects in both val and test: {sorted(val_s & test_s)}")
    unknown = (val_s | test_s) - set(subjects)
    if unknown:
        raise ValueError(f"Unknown subjects in split config: {sorted(unknown)}")
    train = [s for s in subjects if s not in val_s | test_s]
    split = {"train": train, "val": sorted(val_s), "test": sorted(test_s)}
    assert_disjoint(split)
    return split


def assert_disjoint(split: dict[str, list[str]]) -> None:
    names = list(split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(split[a]) & set(split[b])
            if overlap:
                raise AssertionError(f"Subject leakage between {a} and {b}: {sorted(overlap)}")
