import numpy as np
import onnx
import pytest
import torch

from hsi.export import export_onnx, ort_logits, parity, session, to_fp16, to_int8
from hsi.models import DenseWrapper, build_model

SMALL = {
    "pixel_mlp": {"hidden": (16, 16)},
    "patch_cnn": {"patch_size": 5, "spectral_channels": 8, "width": 8},
    "unet": {"base": 4, "depth": 2, "stem_channels": 8},
}


@pytest.mark.parametrize("name", list(SMALL))
def test_onnx_parity_and_dynamic_shape(name, tmp_path):
    torch.manual_seed(0)
    m = build_model(name, 16, 5, **SMALL[name]).eval()
    p = export_onnx(m, 16, tmp_path / "m.onnx", {"classes": ["a", "b", "c", "d", "e"]})
    meta = {kv.key: kv.value for kv in onnx.load(str(p)).metadata_props}
    assert "classes" in meta
    sess = session(p)
    for h, w in [(240, 320), (32, 48)]:  # dynamic H, W
        x = np.random.default_rng(0).standard_normal((1, 16, h, w)).astype(np.float32)
        with torch.no_grad():
            ref = DenseWrapper(m)(torch.from_numpy(x)).numpy()
        out = ort_logits(sess, x)
        assert out.shape == (1, 5, h, w)
        par = parity(ref, out)
        assert par["max_abs_diff"] < 1e-3 and par["argmax_agreement"] > 0.999


def test_fp16_and_int8_variants_run(tmp_path):
    torch.manual_seed(0)
    m = build_model("pixel_mlp", 16, 5, hidden=(16, 16)).eval()
    p = export_onnx(m, 16, tmp_path / "m.onnx")
    x = np.random.default_rng(0).standard_normal((1, 16, 24, 32)).astype(np.float32)
    ref = ort_logits(session(p), x)
    for conv in (to_fp16, to_int8):
        q = conv(p, tmp_path / f"{conv.__name__}.onnx")
        out = ort_logits(session(q), x)
        assert out.shape == ref.shape and out.dtype == np.float32
        assert parity(ref, out)["argmax_agreement"] > 0.9
