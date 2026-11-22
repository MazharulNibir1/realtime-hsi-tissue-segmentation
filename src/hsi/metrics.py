"""Pixel-level metrics from a confusion matrix (only annotated pixels are counted).

Note: per-class pixel Dice = 2TP / (2TP + FP + FN) is identical to per-class pixel F1.
"""

from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, n_classes: int, ignore: int = 255
) -> torch.Tensor:
    """(K, K) int64 counts; rows = true class, cols = predicted class."""
    m = target != ignore
    idx = target[m].long() * n_classes + pred[m].long()
    return torch.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes).cpu()


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """F1 per class; NaN for classes absent from both targets and predictions."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def summarize(cm: np.ndarray) -> dict:
    cm = np.asarray(cm)
    f1 = per_class_f1(cm)
    present = cm.sum(1) > 0  # macro average over classes present in the targets
    return {
        "macro_f1": float(np.nanmean(f1[present])) if present.any() else float("nan"),
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "per_class_f1": f1.tolist(),
        "n_pixels": int(cm.sum()),
    }
