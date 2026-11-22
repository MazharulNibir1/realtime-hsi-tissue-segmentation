"""Input transform shared by training, ONNX demos and the browser port.

Pipeline for a spectrum x (100 channels, already L1-normalised by the dataset):

    1. y = W @ x                  band projection (identity for full spectrum, 16xN for sensor)
    2. y = y / max(sum|y|, 1e-8)  L1 re-normalisation
    3. mode:
         "l1"           -> y
         "l1_standardize" -> (y - mean) / std   per band, mean/std fitted on training pixels
         "l1_snv"       -> (y - mean(y)) / max(std(y), 1e-8)   per pixel (standard normal variate)

Everything is float32. `to_dict()` holds all parameters needed to reproduce this exactly
in another runtime (see web/ port), including W.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hsi.data.spectral import L1_EPS, band_centres, project_bands

MODES = ("l1", "l1_standardize", "l1_snv")
SNV_EPS = 1e-8
STD_EPS = 1e-8


@dataclass
class SpectralTransform:
    W: np.ndarray  # (n_bands, 100)
    mode: str = "l1"
    mean: np.ndarray | None = field(default=None)
    std: np.ndarray | None = field(default=None)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown normalisation mode {self.mode!r}; choose from {MODES}")
        self.W = np.asarray(self.W, dtype=np.float32)

    @property
    def n_bands(self) -> int:
        return self.W.shape[0]

    @property
    def needs_fit(self) -> bool:
        return self.mode == "l1_standardize" and self.mean is None

    def _project_l1(self, x: np.ndarray) -> np.ndarray:
        is_identity = self.W.shape[0] == self.W.shape[1] and np.array_equal(
            self.W, np.eye(self.W.shape[0], dtype=np.float32)
        )
        y = np.asarray(x, dtype=np.float32) if is_identity else project_bands(x, self.W)
        return y / np.maximum(np.abs(y).sum(axis=-1, keepdims=True), L1_EPS)

    def fit(self, spectra: np.ndarray) -> SpectralTransform:
        """Fit per-band mean/std on (N, 100) training spectra (only used by l1_standardize)."""
        if self.mode == "l1_standardize":
            y = self._project_l1(spectra).astype(np.float64)
            self.mean = y.mean(axis=0).astype(np.float32)
            self.std = np.maximum(y.std(axis=0), STD_EPS).astype(np.float32)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(..., 100) -> (..., n_bands) float32."""
        y = self._project_l1(x)
        if self.mode == "l1_standardize":
            if self.mean is None:
                raise RuntimeError("SpectralTransform(l1_standardize) used before fit()")
            y = (y - self.mean) / self.std
        elif self.mode == "l1_snv":
            mu = y.mean(axis=-1, keepdims=True)
            sd = y.std(axis=-1, keepdims=True)
            y = (y - mu) / np.maximum(sd, SNV_EPS)
        return y.astype(np.float32, copy=False)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "n_bands": self.n_bands,
            "band_centres_nm": band_centres(self.W).round(2).tolist(),
            "W": self.W.tolist(),
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "l1_eps": L1_EPS,
            "snv_eps": SNV_EPS,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpectralTransform:
        f = lambda v: None if v is None else np.asarray(v, dtype=np.float32)  # noqa: E731
        return cls(W=f(d["W"]), mode=d["mode"], mean=f(d.get("mean")), std=f(d.get("std")))


def transform_from_config(band_cfg: dict, mode: str) -> SpectralTransform:
    """Build an (unfitted) transform from a `bands.<setting>` config entry."""
    from hsi.data.spectral import band_matrix

    W = band_matrix(band_cfg["method"], band_cfg.get("n_bands", 16), band_cfg.get("fwhm_nm", 20.0))
    return SpectralTransform(W=W, mode=mode)
