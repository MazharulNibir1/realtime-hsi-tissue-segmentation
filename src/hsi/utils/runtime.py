"""Device selection (MPS -> CPU; no CUDA assumed) and reproducible seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return the compute device.

    prefer: "auto" (MPS if available, else CPU), "mps", or "cpu".
    """
    if prefer not in {"auto", "mps", "cpu"}:
        raise ValueError(f"Unknown device preference: {prefer!r}")
    if prefer in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "mps":
        raise RuntimeError("MPS requested but not available")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + MPS)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (global seed for libs using legacy RNG)
    torch.manual_seed(seed)  # also seeds MPS generator
