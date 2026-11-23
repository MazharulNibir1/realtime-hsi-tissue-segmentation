import json
from pathlib import Path

import numpy as np
import pytest

from hsi.data.spectral import WAVELENGTHS_NM
from hsi.samples import (
    CLASS_COLOURS,
    pseudo_rgb_bands,
    pseudo_rgb_params,
    read_sample,
    render_pseudo_rgb,
    write_sample,
)

ROOT = Path(__file__).resolve().parents[1]


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    cube = rng.random((6, 8, 100)).astype(np.float16)
    labels = np.full((6, 8), 255, np.uint8)
    labels[2:4, 3:5] = 7
    write_sample(tmp_path, "s", cube, labels, {"organ": "spleen"})
    c2, l2, meta = read_sample(tmp_path, "s")
    np.testing.assert_array_equal(c2, cube)
    np.testing.assert_array_equal(l2, labels)
    assert meta["shape"] == [6, 8, 100] and meta["organ"] == "spleen"
    assert (tmp_path / "s.cube.bin").stat().st_size == 6 * 8 * 100 * 2


def test_palette_and_rgb_bands():
    assert len(CLASS_COLOURS) == 20 and len(set(CLASS_COLOURS)) == 20
    r, g, b = pseudo_rgb_bands(WAVELENGTHS_NM)
    assert r > g > b


@pytest.mark.parametrize("d", ["app/samples", "web/public/samples"])
def test_published_samples_consistent(d):
    """If samples were generated, both demo copies are identical and readable."""
    a, b = ROOT / "app/samples", ROOT / d
    if not (a / "index.json").exists():
        pytest.skip("samples not generated")
    index = json.loads((b / "index.json").read_text())
    for s in index["samples"]:
        cube, labels, meta = read_sample(b, s["id"])
        assert cube.shape[2] == 100 and labels.shape == cube.shape[:2]
        np.testing.assert_allclose(np.abs(cube.astype(np.float32)).sum(-1), 1, atol=5e-3)
        assert (a / meta["cube_file"]).read_bytes() == (b / meta["cube_file"]).read_bytes()


def test_pseudo_rgb_render():
    cube = np.random.default_rng(0).random((10, 12, 100)).astype(np.float32)
    p = pseudo_rgb_params(cube, [80, 14, 2])
    img = render_pseudo_rgb(cube, p)
    assert img.shape == (10, 12, 3) and img.dtype == np.uint8
    assert img.min() == 0 and img.max() == 255
