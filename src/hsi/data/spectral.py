"""Spectral operations: wavelengths, spatial binning, L1 normalisation, 16-band sensor simulation.

All band operations are linear, non-negative projections `y = x @ W.T` with W of shape
(n_out, 100). Because L1 normalisation only rescales each pixel, projecting already-L1
spectra and re-normalising gives exactly L1(W @ reflectance):
    L1(W (x/|x|)) = W x / |W x| = L1(W x)    for x >= 0, W >= 0.
So simulating the sensor from the dataset's L1 cubes is equivalent to simulating it from
reflectance (up to the few slightly negative values in the source data).

Not simulated: the spatial mosaic itself (each sensor pixel seeing one band followed by
demosaicking). The 2x2 binning to 240x320 only approximates the reduced spatial resolution.
"""

from __future__ import annotations

import numpy as np

N_CHANNELS = 100
# HeiPorSPECTRAL / TIVITA: 100 channels, 500-1000 nm (same convention as the htc library).
WAVELENGTHS_NM = np.linspace(500.0, 1000.0, N_CHANNELS)

L1_EPS = 1e-8


def l1_normalize(x: np.ndarray, eps: float = L1_EPS) -> np.ndarray:
    """x / sum(|x|) over the last axis, in float32."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.abs(x).sum(axis=-1, keepdims=True), eps)


def bin_spatial(cube: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool (H, W, C) by `factor` in both spatial dims (crops any remainder), float32."""
    if factor == 1:
        return np.asarray(cube, dtype=np.float32)
    h, w, c = cube.shape
    h2, w2 = h // factor, w // factor
    x = np.asarray(cube[: h2 * factor, : w2 * factor], dtype=np.float32)
    return x.reshape(h2, factor, w2, factor, c).mean(axis=(1, 3))


def downsample_labels(labels: np.ndarray, factor: int, ignore: int = 255) -> np.ndarray:
    """Keep a label only where all factor x factor sub-pixels agree; otherwise `ignore`."""
    if factor == 1:
        return labels.copy()
    h, w = labels.shape
    h2, w2 = h // factor, w // factor
    blocks = labels[: h2 * factor, : w2 * factor].reshape(h2, factor, w2, factor)
    first = blocks[:, :1, :, :1]
    pure = (blocks == first).all(axis=(1, 3))
    out = np.where(pure, first[:, 0, :, 0], ignore)
    return out.astype(np.uint8)


# --------------------------------------------------------------------------- band simulation


def band_matrix(method: str, n_bands: int = 16, fwhm_nm: float = 20.0) -> np.ndarray:
    """Projection matrix (n_bands, 100) simulating a snapshot sensor. Rows sum to 1.

    method:
      "identity" - full spectrum (100 x 100 identity), n_bands ignored
      "box"      - n_bands equal-width bins over 500-1000 nm; average channels per bin
      "gaussian" - n_bands evenly spaced centres, Gaussian response with the given FWHM
                   (closer to Fabry-Perot filter mosaics; neighbouring bands may overlap)
      "select"   - pick the single channel nearest to each bin centre
    """
    wl = WAVELENGTHS_NM
    if method == "identity":
        return np.eye(N_CHANNELS, dtype=np.float32)
    edges = np.linspace(wl[0], wl[-1], n_bands + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    W = np.zeros((n_bands, N_CHANNELS), dtype=np.float64)
    if method == "box":
        idx = np.clip(np.searchsorted(edges, wl, side="right") - 1, 0, n_bands - 1)
        W[idx, np.arange(N_CHANNELS)] = 1.0
    elif method == "gaussian":
        sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
        W = np.exp(-0.5 * ((wl[None, :] - centres[:, None]) / sigma) ** 2)
    elif method == "select":
        W[np.arange(n_bands), np.abs(wl[None, :] - centres[:, None]).argmin(axis=1)] = 1.0
    else:
        raise ValueError(f"Unknown band method: {method!r}")
    if (W.sum(axis=1) == 0).any():
        raise ValueError("A simulated band has no channels; use fewer bands")
    return (W / W.sum(axis=1, keepdims=True)).astype(np.float32)


def band_centres(W: np.ndarray) -> np.ndarray:
    """Response-weighted centre wavelength of each simulated band (nm)."""
    return (W @ WAVELENGTHS_NM).astype(np.float32)


def project_bands(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(..., 100) -> (..., n_bands), float32."""
    return np.asarray(x, dtype=np.float32) @ W.T
