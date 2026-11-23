"""Demo sample format shared by the Gradio app and the browser demo.

<id>.cube.bin    float16, little-endian, row-major (H, W, C) = L1-normalised spectra (C = 100)
<id>.labels.bin  uint8 (H, W): annotated class index, 255 = not annotated
<id>.json        metadata (shape, dtype, wavelengths, label, provenance, rgb bands ...)
index.json       list of samples + class names/colours + class mean spectra + licence

Model input is produced from the cube by the run's SpectralTransform (transform.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CUBE_DTYPE = "<f2"
LABEL_DTYPE = "u1"

# 20 high-contrast colours (Kelly's maximum-contrast set without white/black), fixed per class index.
CLASS_COLOURS = [
    "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032", "#C2B280", "#848482", "#008856",
    "#E68FAC", "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C", "#DCD300", "#882D17",
    "#8DB600", "#654522", "#E25822", "#2B3D26",
]  # fmt: skip


def pseudo_rgb_bands(wavelengths: np.ndarray, targets=(650.0, 570.0, 510.0)) -> list[int]:
    """Channel indices closest to red/green/blue-ish targets (the camera starts at 500 nm)."""
    return [int(np.abs(wavelengths - t).argmin()) for t in targets]


def pseudo_rgb_params(cube: np.ndarray, bands: list[int], gamma: float = 1.2) -> dict:
    """Display recipe: per-channel 1st-99th percentile stretch, then x ** (1 / gamma).

    Stored with each sample so both demos render identically without porting percentiles.
    """
    rgb = np.asarray(cube[..., bands], dtype=np.float32)
    lo = np.percentile(rgb, 1, axis=(0, 1))
    hi = np.percentile(rgb, 99, axis=(0, 1))
    return {"bands": bands, "lo": lo.round(7).tolist(), "hi": hi.round(7).tolist(), "gamma": gamma}


def render_pseudo_rgb(cube: np.ndarray, p: dict) -> np.ndarray:
    """(H, W, C) cube -> (H, W, 3) uint8 using a pseudo_rgb_params recipe."""
    rgb = np.asarray(cube[..., p["bands"]], dtype=np.float32)
    lo, hi = np.asarray(p["lo"], np.float32), np.asarray(p["hi"], np.float32)
    x = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-12), 0, 1) ** (1 / p["gamma"])
    return (x * 255 + 0.5).astype(np.uint8)


def write_sample(
    out_dir: Path, sample_id: str, cube: np.ndarray, labels: np.ndarray, meta: dict
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w, c = cube.shape
    assert labels.shape == (h, w)
    (out_dir / f"{sample_id}.cube.bin").write_bytes(
        np.ascontiguousarray(cube, dtype=CUBE_DTYPE).tobytes()
    )
    (out_dir / f"{sample_id}.labels.bin").write_bytes(
        np.ascontiguousarray(labels, dtype=LABEL_DTYPE).tobytes()
    )
    meta = {
        "id": sample_id,
        "cube_file": f"{sample_id}.cube.bin",
        "labels_file": f"{sample_id}.labels.bin",
        "shape": [h, w, c],
        "layout": "HWC",
        "cube_dtype": "float16-le",
        "labels_dtype": "uint8",
        "ignore_index": 255,
        **meta,
    }
    (out_dir / f"{sample_id}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def read_sample(sample_dir: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = json.loads((sample_dir / f"{sample_id}.json").read_text())
    h, w, c = meta["shape"]
    cube = np.frombuffer((sample_dir / meta["cube_file"]).read_bytes(), dtype=CUBE_DTYPE).reshape(
        h, w, c
    )
    labels = np.frombuffer(
        (sample_dir / meta["labels_file"]).read_bytes(), dtype=LABEL_DTYPE
    ).reshape(h, w)
    return cube, labels, meta
