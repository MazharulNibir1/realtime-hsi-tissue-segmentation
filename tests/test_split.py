import json
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from hsi.data.split import assert_disjoint, make_split

ROOT = Path(__file__).resolve().parents[1]


def test_make_split_disjoint():
    s = make_split([f"P{i}" for i in range(10)], val=["P1", "P2"], test=["P3"])
    assert s["val"] == ["P1", "P2"] and s["test"] == ["P3"] and len(s["train"]) == 7
    assert_disjoint(s)


def test_overlap_rejected():
    with pytest.raises(ValueError):
        make_split(["A", "B", "C"], val=["A"], test=["A"])
    with pytest.raises(ValueError):
        make_split(["A", "B"], val=["Z"], test=["B"])
    with pytest.raises(AssertionError):
        assert_disjoint({"train": ["A", "B"], "test": ["B"]})


def test_configured_split_on_manifest():
    """The real split: no pig overlap, all images assigned, every organ in every split."""
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    images = json.loads((ROOT / cfg["paths"]["manifest"]).read_text())["images"]
    split = make_split({im["subject"] for im in images}, cfg["split"]["val"], cfg["split"]["test"])
    assert sum(len(v) for v in split.values()) == len({im["subject"] for im in images})

    organs_by_part = defaultdict(set)
    pigs_per_organ_train = defaultdict(set)
    for im in images:
        part = next(p for p, subs in split.items() if im["subject"] in subs)
        organs_by_part[part].add(im["label"])
        if part == "train":
            pigs_per_organ_train[im["label"]].add(im["subject"])
    all_organs = {im["label"] for im in images}
    assert len(all_organs) == 20
    for part in ("train", "val", "test"):
        assert organs_by_part[part] == all_organs, part
    assert min(len(v) for v in pigs_per_organ_train.values()) >= 4
