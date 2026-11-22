import numpy as np
import pytest

from hsi.data.spectral import (
    WAVELENGTHS_NM,
    band_centres,
    band_matrix,
    bin_spatial,
    downsample_labels,
    l1_normalize,
    project_bands,
)


def test_wavelengths():
    assert WAVELENGTHS_NM.shape == (100,)
    assert WAVELENGTHS_NM[0] == 500 and WAVELENGTHS_NM[-1] == 1000


def test_bin_spatial_shape_and_mean():
    x = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    y = bin_spatial(x, 2)
    assert y.shape == (2, 3, 3)
    np.testing.assert_allclose(y[0, 0], x[:2, :2].reshape(-1, 3).mean(0))


def test_downsample_labels_keeps_only_pure_blocks():
    lab = np.full((4, 4), 255, np.uint8)
    lab[:2, :2] = 3  # pure block
    lab[2:, 2:] = 5
    lab[3, 3] = 255  # impure block
    out = downsample_labels(lab, 2)
    assert out.tolist() == [[3, 255], [255, 255]]


@pytest.mark.parametrize("method", ["box", "gaussian", "select"])
def test_band_matrix_16(method):
    W = band_matrix(method, 16)
    assert W.shape == (16, 100)
    np.testing.assert_allclose(W.sum(1), 1, rtol=1e-6)
    assert (W >= 0).all()
    c = band_centres(W)
    assert (np.diff(c) > 0).all() and c[0] >= 500 and c[-1] <= 1000


def test_box_uses_every_channel_once():
    W = band_matrix("box", 16)
    assert ((W > 0).sum(0) == 1).all()


def test_identity():
    W = band_matrix("identity")
    x = np.random.default_rng(0).random((5, 100)).astype(np.float32)
    np.testing.assert_allclose(project_bands(x, W), x, rtol=1e-6)


def test_projection_commutes_with_l1():
    """Simulating the sensor from L1 data == L1 of the sensor applied to reflectance."""
    rng = np.random.default_rng(0)
    refl = rng.random((50, 100)).astype(np.float32) + 0.01
    W = band_matrix("box", 16)
    a = l1_normalize(project_bands(l1_normalize(refl), W))
    b = l1_normalize(project_bands(refl, W))
    np.testing.assert_allclose(a, b, rtol=1e-5)
