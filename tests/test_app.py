"""The Gradio app's self-contained inference must match the training-side code exactly."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hsi.data.normalize import SpectralTransform

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
pytestmark = pytest.mark.skipif(not (APP / "models").exists(), reason="demo models not prepared")


@pytest.fixture(scope="module")
def inference():
    spec = importlib.util.spec_from_file_location("app_inference", APP / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app_inference"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("bands", ["full", "snapshot16"])
def test_transform_port_matches_training_code(inference, bands):
    d = json.loads((APP / f"models/{bands}/transform.json").read_text())
    x = np.random.default_rng(0).random((7, 9, 100)).astype(np.float32)
    x /= x.sum(-1, keepdims=True)
    np.testing.assert_allclose(
        inference.Transform(d)(x), SpectralTransform.from_dict(d)(x), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("bands", ["full", "snapshot16"])
def test_sample_accuracy_matches_metadata(inference, bands):
    model = inference.DemoModel(APP / f"models/{bands}")
    for s in inference.load_index()["samples"]:
        cube, labels, meta = inference.load_sample(s["id"])
        probs, _ = model.predict(cube)
        m = labels != inference.IGNORE
        acc = float((probs.argmax(-1)[m] == labels[m]).mean())
        assert acc == pytest.approx(meta["demo_model_pixel_accuracy"][bands], abs=1e-9), s["id"]


def test_user_cube_validation(inference):
    with pytest.raises(ValueError):
        inference.load_user_cube(np.zeros((4, 4, 16)))
    cube = inference.load_user_cube(np.full((4, 4, 100), 3.0))
    np.testing.assert_allclose(cube.sum(-1), 1, rtol=1e-6)


def test_overlay_and_outline(inference):
    rgb = np.zeros((6, 6, 3), np.uint8)
    pred = np.zeros((6, 6), np.int64)
    gt = np.zeros((6, 6), bool)
    gt[1:5, 1:5] = True
    out = inference.overlay(rgb, pred, ["#ff0000"] * 20, 1.0, gt)
    assert out[1, 1].tolist() == [255, 255, 255]  # outline
    assert out[2, 2].tolist() == [255, 0, 0]  # interior keeps class colour
