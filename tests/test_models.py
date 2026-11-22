import numpy as np
import pytest
import torch

from hsi.metrics import confusion_matrix, per_class_f1, summarize
from hsi.models import DenseWrapper, build_model, count_params
from hsi.train import apply_overrides, class_weights

K = 20
CONFIGS = {
    "pixel_mlp": {"hidden": (128, 128, 64)},
    "patch_cnn": {"patch_size": 11, "spectral_channels": 32, "width": 64},
    "unet": {"base": 16, "depth": 3, "stem_channels": 32},
}


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("bands", [100, 16])
def test_dense_forward_shape_and_param_budget(name, bands):
    torch.manual_seed(0)
    m = build_model(name, bands, K, **CONFIGS[name]).eval()
    assert count_params(m) < 2_000_000
    with torch.no_grad():
        y = DenseWrapper(m)(torch.randn(2, bands, 240, 320))
    assert y.shape == (2, K, 240, 320)


def test_training_input_shapes():
    mlp = build_model("pixel_mlp", 16, K)
    assert mlp(torch.randn(5, 16)).shape == (5, K)
    cnn = build_model("patch_cnn", 16, K, patch_size=11)
    assert cnn(torch.randn(5, 16, 11, 11)).shape == (5, K)
    unet = build_model("unet", 16, K)
    assert unet(torch.randn(1, 16, 48, 64)).shape == (1, K, 48, 64)


def test_patch_cnn_dense_equals_per_patch():
    """One dense pass == classifying each pixel's reflect-padded 11x11 patch."""
    torch.manual_seed(0)
    m = build_model("patch_cnn", 8, K, patch_size=11, spectral_channels=8, width=8).eval()
    x = torch.randn(1, 8, 16, 20)
    with torch.no_grad():
        dense = m.forward_dense(x)
        padded = torch.nn.functional.pad(x, (5,) * 4, mode="reflect")
        for r, c in [(0, 0), (7, 9), (15, 19), (3, 18)]:
            patch = padded[:, :, r : r + 11, c : c + 11]
            torch.testing.assert_close(m(patch)[0], dense[0, :, r, c], rtol=1e-4, atol=1e-5)


def test_metrics():
    target = torch.tensor([0, 0, 1, 1, 2, 255])
    pred = torch.tensor([0, 1, 1, 1, 0, 2])
    cm = confusion_matrix(pred, target, 3).numpy()
    assert cm.sum() == 5  # ignored pixel excluded
    f1 = per_class_f1(cm)
    np.testing.assert_allclose(f1, [0.5, 0.8, 0.0])
    s = summarize(cm)
    assert s["macro_f1"] == pytest.approx(np.mean([0.5, 0.8, 0.0]))
    assert s["accuracy"] == pytest.approx(3 / 5)


def test_class_weights_and_overrides():
    w = class_weights(np.array([100, 400]), "sqrt_inverse")
    assert w[0] / w[1] == pytest.approx(2.0) and w.mean() == pytest.approx(1.0)
    cfg = apply_overrides({"train": {"epochs": 30}}, ["train.epochs=1", "model.hidden=[8, 8]"])
    assert cfg["train"]["epochs"] == 1 and cfg["model"]["hidden"] == [8, 8]


def test_lr_schedule():
    from hsi.train import lr_at

    assert lr_at(0.0, 1.0) < 0.05  # warm-up starts low
    assert lr_at(0.05, 1.0) == pytest.approx(1.0, rel=0.02)  # peak after warm-up
    assert lr_at(1.0, 1.0) == pytest.approx(0.01)  # cosine floor
    assert lr_at(0.5, 1.0) == pytest.approx(0.505, rel=0.01)
