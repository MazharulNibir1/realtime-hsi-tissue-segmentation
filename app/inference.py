"""Inference + rendering for the Gradio demo. Self-contained: NumPy + ONNX Runtime only.

Input transform (identical to hsi.data.normalize.SpectralTransform; parity-tested):
    x : (H, W, 100) L1-normalised spectra
    y = x @ W.T                           band projection (identity or 16 x 100 box filter)
    y = y / max(sum(|y|), l1_eps)         per-pixel L1 re-normalisation
    y = (y - mean) / std                  per-band standardisation (training-set statistics)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
MODELS = HERE / "models"
IGNORE = 255


class Transform:
    def __init__(self, d: dict):
        self.mode = d["mode"]
        self.W = np.asarray(d["W"], dtype=np.float32)
        self.mean = None if d.get("mean") is None else np.asarray(d["mean"], dtype=np.float32)
        self.std = None if d.get("std") is None else np.asarray(d["std"], dtype=np.float32)
        self.l1_eps = float(d.get("l1_eps", 1e-8))
        self.snv_eps = float(d.get("snv_eps", 1e-8))
        self.n_bands = self.W.shape[0]
        self.band_centres = np.asarray(d["band_centres_nm"], dtype=np.float32)
        self.identity = self.W.shape[0] == self.W.shape[1] and np.allclose(
            self.W, np.eye(self.n_bands)
        )

    def project_l1(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = x if self.identity else x @ self.W.T
        return y / np.maximum(np.abs(y).sum(axis=-1, keepdims=True), self.l1_eps)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = self.project_l1(x)
        if self.mode == "l1_standardize":
            y = (y - self.mean) / self.std
        elif self.mode == "l1_snv":
            y = (y - y.mean(-1, keepdims=True)) / np.maximum(y.std(-1, keepdims=True), self.snv_eps)
        return y.astype(np.float32, copy=False)


class DemoModel:
    def __init__(self, model_dir: Path, threads: int = 0):
        self.info = json.loads((model_dir / "model.json").read_text())
        self.transform = Transform(json.loads((model_dir / "transform.json").read_text()))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"]
        )

    def predict(self, cube: np.ndarray) -> tuple[np.ndarray, dict]:
        """cube (H, W, 100) -> softmax probabilities (H, W, K), timings in ms."""
        t0 = time.perf_counter()
        x = np.ascontiguousarray(self.transform(cube).transpose(2, 0, 1)[None])
        t1 = time.perf_counter()
        logits = self.session.run(["logits"], {"input": x})[0][0]
        t2 = time.perf_counter()
        z = logits - logits.max(0, keepdims=True)
        e = np.exp(z)
        probs = (e / e.sum(0, keepdims=True)).transpose(1, 2, 0)
        return probs, {"preprocess_ms": (t1 - t0) * 1e3, "inference_ms": (t2 - t1) * 1e3}


def load_models() -> dict[str, DemoModel]:
    return {d.name: DemoModel(d) for d in sorted(MODELS.iterdir()) if (d / "model.onnx").exists()}


def load_index() -> dict:
    return json.loads((SAMPLES / "index.json").read_text())


def load_sample(sample_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = json.loads((SAMPLES / f"{sample_id}.json").read_text())
    h, w, c = meta["shape"]
    cube = np.frombuffer((SAMPLES / meta["cube_file"]).read_bytes(), dtype="<f2").reshape(h, w, c)
    labels = np.frombuffer((SAMPLES / meta["labels_file"]).read_bytes(), dtype="u1").reshape(h, w)
    return cube.astype(np.float32), labels.copy(), meta


def load_user_cube(arr: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Validate an uploaded (H, W, 100) cube; L1-normalise each pixel (reflectance or L1 input)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 100:
        raise ValueError(
            f"Expected shape (H, W, 100) for 100 bands at 500-1000 nm; got {arr.shape}"
        )
    if max(arr.shape[:2]) > max_side:
        raise ValueError(
            f"Image too large for this demo (max {max_side} px per side); got {arr.shape[:2]}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("Cube contains NaN or Inf values")
    return arr / np.maximum(np.abs(arr).sum(-1, keepdims=True), 1e-8)


def pseudo_rgb(cube: np.ndarray, recipe: dict | None, wavelengths: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8. Uses the sample's stored recipe, or per-channel percentiles for uploads."""
    if recipe is None:
        bands = [int(np.abs(wavelengths - t).argmin()) for t in (650.0, 570.0, 510.0)]
        rgb = cube[..., bands]
        lo, hi, gamma = np.percentile(rgb, 1, (0, 1)), np.percentile(rgb, 99, (0, 1)), 1.2
    else:
        rgb = cube[..., recipe["bands"]]
        lo, hi, gamma = np.asarray(recipe["lo"]), np.asarray(recipe["hi"]), recipe["gamma"]
    x = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-12), 0, 1) ** (1 / gamma)
    return (x * 255 + 0.5).astype(np.uint8)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a boolean mask (4-neighbourhood)."""
    m = np.pad(mask, 1)
    interior = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return mask & ~interior


def overlay(
    rgb: np.ndarray,
    pred: np.ndarray,
    colours: list[str],
    alpha: float,
    gt: np.ndarray | None,
    conf: np.ndarray | None = None,
    min_conf: float = 0.0,
) -> np.ndarray:
    """Blend class colours over the image; pixels with max probability < min_conf stay uncoloured."""
    palette = np.array([hex_to_rgb(c) for c in colours], dtype=np.float32)
    out = (1 - alpha) * rgb.astype(np.float32) + alpha * palette[pred]
    if conf is not None and min_conf > 0:
        low = conf < min_conf
        out[low] = rgb[low]
    if gt is not None and gt.any():
        out[mask_outline(gt)] = 255.0  # annotated region (ground truth) outlined in white
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def upscale(img: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(np.repeat(img, k, 0), k, 1)
