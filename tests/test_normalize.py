import numpy as np
import pytest

from hsi.data.normalize import SpectralTransform, transform_from_config
from hsi.data.spectral import band_matrix


@pytest.fixture
def spectra():
    return np.random.default_rng(0).random((200, 100)).astype(np.float32)


@pytest.mark.parametrize(
    "setting,n", [({"method": "identity"}, 100), ({"method": "box", "n_bands": 16}, 16)]
)
@pytest.mark.parametrize("mode", ["l1", "l1_standardize", "l1_snv"])
def test_shapes_and_modes(spectra, setting, n, mode):
    t = transform_from_config(setting, mode).fit(spectra)
    y = t(spectra.reshape(10, 20, 100))
    assert y.shape == (10, 20, n) and y.dtype == np.float32
    flat = y.reshape(-1, n)
    if mode == "l1":
        np.testing.assert_allclose(np.abs(flat).sum(1), 1, rtol=1e-5)
    elif mode == "l1_standardize":
        np.testing.assert_allclose(flat.mean(0), 0, atol=1e-4)
        np.testing.assert_allclose(flat.std(0), 1, atol=1e-3)
    else:
        np.testing.assert_allclose(flat.mean(1), 0, atol=1e-5)
        np.testing.assert_allclose(flat.std(1), 1, atol=1e-3)


def test_standardize_requires_fit(spectra):
    t = SpectralTransform(band_matrix("box", 16), "l1_standardize")
    with pytest.raises(RuntimeError):
        t(spectra)


def test_dict_roundtrip(spectra):
    t = SpectralTransform(band_matrix("gaussian", 16), "l1_standardize").fit(spectra)
    t2 = SpectralTransform.from_dict(t.to_dict())
    np.testing.assert_array_equal(t(spectra), t2(spectra))


def test_unknown_mode():
    with pytest.raises(ValueError):
        SpectralTransform(band_matrix("identity"), "zscore")
