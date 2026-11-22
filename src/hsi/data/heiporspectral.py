"""HeiPorSPECTRAL raw subset -> processed arrays.

Raw (from scripts/download_data.py, under data/raw/HeiPorSPECTRAL/):
  intermediates/preprocessing/L1/<image>.blosc   float16 (480, 640, 100), L1-normalised
  data/subjects/<pig>/<ts>/annotations/<ts>#polygon#<annotator>#<organ>#binary.png

Processed (under data/processed/):
  cubes/<image>.npy                 float16 (H/b, W/b, 100), 2x2 mean-binned then L1 re-normalised
  labels/<source>/<image>.npy       uint8 (H/b, W/b): class index, 255 = unlabelled
                                    source = annotator1 | majority (>= 2 of 3 annotators)
  index.json                        per-image metadata + class names
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from hsi.data.blosc_io import load_blosc
from hsi.data.spectral import bin_spatial, downsample_labels, l1_normalize

IGNORE = 255
LABEL_SOURCES = ("annotator1", "majority")


def load_class_names(dataset_settings: Path) -> list[str]:
    """Class names ordered by label index (0..last_valid_label_index)."""
    s = json.loads(Path(dataset_settings).read_text())
    mapping = s["label_mapping"]
    n = s["last_valid_label_index"] + 1
    names = [None] * n
    for name, idx in mapping.items():
        if idx < n:
            names[idx] = name
    assert all(names), "label mapping has gaps"
    return names


def raw_paths(raw_root: Path, entry: dict, annotators: list[str]) -> tuple[Path, dict[str, Path]]:
    cube = raw_root / "intermediates/preprocessing/L1" / f"{entry['image_name']}.blosc"
    ann_dir = raw_root / "data/subjects" / entry["subject"] / entry["timestamp"] / "annotations"
    masks = {
        a: ann_dir / f"{entry['timestamp']}#polygon#{a}#{entry['label']}#binary.png"
        for a in annotators
    }
    return cube, masks


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)) > 0


def build_labels(masks: dict[str, np.ndarray], label_idx: int) -> dict[str, np.ndarray]:
    """Full-resolution uint8 label maps for each label source."""
    a1 = masks["annotator1"]
    votes = sum(m.astype(np.uint8) for m in masks.values())
    out = {
        "annotator1": np.where(a1, label_idx, IGNORE).astype(np.uint8),
        "majority": np.where(votes >= 2, label_idx, IGNORE).astype(np.uint8),
    }
    return out


def process_cube(cube: np.ndarray, factor: int) -> np.ndarray:
    """(480, 640, 100) L1 cube -> binned, L1 re-normalised float16."""
    return l1_normalize(bin_spatial(cube, factor)).astype(np.float16)


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def process_image(
    entry: dict, raw_root: Path, out_dir: Path, label_idx: int, factor: int, annotators: list[str]
) -> dict:
    """Process one image (idempotent: skips if all outputs exist). Returns index metadata."""
    name = entry["image_name"]
    cube_out = out_dir / "cubes" / f"{name}.npy"
    label_outs = {s: out_dir / "labels" / s / f"{name}.npy" for s in LABEL_SOURCES}
    if not (cube_out.exists() and all(p.exists() for p in label_outs.values())):
        cube_path, mask_paths = raw_paths(raw_root, entry, annotators)
        cube = load_blosc(cube_path)
        if cube.shape != (480, 640, 100):
            raise ValueError(f"{name}: unexpected cube shape {cube.shape}")
        labels = build_labels({a: read_mask(p) for a, p in mask_paths.items()}, label_idx)
        for s, p in label_outs.items():
            _atomic_save(p, downsample_labels(labels[s], factor, IGNORE))
        _atomic_save(cube_out, process_cube(cube, factor))
    meta = {
        k: entry[k]
        for k in ("image_name", "subject", "timestamp", "label", "situs", "angle", "repetition")
    }
    meta["label_idx"] = label_idx
    for s, p in label_outs.items():
        meta[f"labelled_px_{s}"] = int((np.load(p) != IGNORE).sum())
    return meta
