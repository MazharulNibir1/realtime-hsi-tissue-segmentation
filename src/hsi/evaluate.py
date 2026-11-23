"""Test-set evaluation of trained runs + class mean spectra.

Protocol (identical for every model): dense full-frame inference on all test-pig images with the
run's best-validation checkpoint; metrics on annotated pixels only.

Metrics
  pixel macro F1 (primary), accuracy, per-class F1
  per-class Dice: with metrics restricted to annotated pixels, pixel Dice 2TP/(2TP+FP+FN) equals
                  pixel F1, so Dice is reported once (as F1) rather than as a second number
  image-level accuracy: majority predicted class inside the annotated polygon == organ
  per-pig macro F1 (each test pig separately)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hsi.data.dataset import ProcessedStore
from hsi.data.heiporspectral import IGNORE
from hsi.data.normalize import SpectralTransform
from hsi.metrics import per_class_f1, summarize
from hsi.models import DenseWrapper
from hsi.train import load_run


class TestSet:
    """Test frames held in RAM once (raw float16 L1 cubes); each run applies its own transform."""

    def __init__(self, store: ProcessedStore, names: list[str]):
        self.names = names
        self.cubes = [np.asarray(store.cube(n)) for n in names]
        self.labels = [store.labels(n) for n in names]
        self.subjects = [store.entries[n]["subject"] for n in names]
        self.organs = [store.entries[n]["label_idx"] for n in names]


@torch.no_grad()
def predict(model, transform: SpectralTransform, cubes, device, batch_size: int = 4):
    dense = DenseWrapper(model).to(device).eval()
    for i in range(0, len(cubes), batch_size):
        x = np.stack([transform(c) for c in cubes[i : i + batch_size]]).transpose(0, 3, 1, 2)
        logits = dense(torch.from_numpy(np.ascontiguousarray(x)).to(device))
        yield from logits.argmax(1).cpu().numpy().astype(np.uint8)


def _cm(pred: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    m = target != IGNORE
    return np.bincount(target[m].astype(np.int64) * k + pred[m], minlength=k * k).reshape(k, k)


def evaluate_run(run_dir: Path, test: TestSet, n_classes: int, device) -> dict:
    model, transform, cfg = load_run(run_dir, device)
    cm = np.zeros((n_classes, n_classes), np.int64)
    cm_pig: dict[str, np.ndarray] = {}
    img_correct = []
    for pred, lab, subj, organ in zip(
        predict(model, transform, test.cubes, device),
        test.labels,
        test.subjects,
        test.organs,
        strict=True,
    ):
        c = _cm(pred, lab, n_classes)
        cm += c
        cm_pig[subj] = cm_pig.get(subj, 0) + c
        inside = pred[lab != IGNORE]
        img_correct.append(int(np.bincount(inside, minlength=n_classes).argmax() == organ))
    out = summarize(cm)
    out.update(
        {
            "run": run_dir.name,
            "model": cfg["model"]["name"],
            "bands": cfg["bands"],
            "seed": cfg["seed"],
            "image_accuracy": float(np.mean(img_correct)),
            "per_pig_macro_f1": {p: summarize(c)["macro_f1"] for p, c in sorted(cm_pig.items())},
            "confusion_matrix": cm.tolist(),
        }
    )
    return out


def class_mean_spectra(store: ProcessedStore, names: list[str], transforms: dict[str, np.ndarray]):
    """Per-class mean/std of L1 spectra over all annotated pixels of `names`.

    transforms: {setting: W} band matrices; spectra are projected and L1 re-normalised per pixel
    (the demo compares a clicked pixel against these, after the same projection).
    """
    k = len(store.class_names)
    acc = {s: [np.zeros((k, W.shape[0])), np.zeros((k, W.shape[0]))] for s, W in transforms.items()}
    count = np.zeros(k, np.int64)
    for n in names:
        lab = store.labels(n)
        m = lab != IGNORE
        if not m.any():
            continue
        cls = int(lab[m][0])  # one organ per image
        x = np.asarray(store.cube(n), dtype=np.float32)[m]
        count[cls] += len(x)
        for s, W in transforms.items():
            y = x @ W.T
            y /= np.maximum(np.abs(y).sum(1, keepdims=True), 1e-8)
            acc[s][0][cls] += y.sum(0)
            acc[s][1][cls] += (y.astype(np.float64) ** 2).sum(0)
    out = {"class_names": store.class_names, "n_pixels": count.tolist(), "settings": {}}
    for s, (sm, sq) in acc.items():
        mean = sm / np.maximum(count[:, None], 1)
        std = np.sqrt(np.maximum(sq / np.maximum(count[:, None], 1) - mean**2, 0))
        out["settings"][s] = {"mean": mean.tolist(), "std": std.tolist()}
    return out


def aggregate(results: list[dict]) -> dict:
    """mean/std over seeds per (model, bands)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        groups.setdefault((r["model"], r["bands"]), []).append(r)
    agg = {}
    for (m, b), rs in sorted(groups.items()):

        def ms(key, rs=rs):
            v = np.array([r[key] for r in rs], dtype=np.float64)
            return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

        cm_sum = np.sum([np.array(r["confusion_matrix"]) for r in rs], axis=0)
        f1_seeds = np.array([r["per_class_f1"] for r in rs], dtype=np.float64)
        pigs = sorted(rs[0]["per_pig_macro_f1"])
        agg[f"{m}/{b}"] = {
            "model": m,
            "bands": b,
            "n_seeds": len(rs),
            "macro_f1": ms("macro_f1"),
            "accuracy": ms("accuracy"),
            "image_accuracy": ms("image_accuracy"),
            "per_class_f1_mean": np.nanmean(f1_seeds, 0).tolist(),
            "per_class_f1_std": np.nanstd(f1_seeds, 0, ddof=1).tolist() if len(rs) > 1 else None,
            "per_pig_macro_f1": {
                p: {
                    "mean": float(np.mean([r["per_pig_macro_f1"][p] for r in rs])),
                    "std": float(np.std([r["per_pig_macro_f1"][p] for r in rs], ddof=1)),
                }
                for p in pigs
            },
            "confusion_matrix_sum": cm_sum.tolist(),
            "pooled_per_class_f1": per_class_f1(cm_sum).tolist(),
        }
    return agg


def top_confusions(cm: np.ndarray, names: list[str], n: int = 8) -> list[tuple[str, str, float]]:
    """Largest off-diagonal row-normalised entries: (true, predicted, fraction of true pixels)."""
    cm = np.asarray(cm, dtype=np.float64)
    r = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    np.fill_diagonal(r, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-r, axis=None)[:n], r.shape))[0]
    return [(names[i], names[j], float(r[i, j])) for i, j in idx]


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
