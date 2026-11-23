"""ONNX export, precision variants, parity checks and latency benchmarks.

Exported graph: DenseWrapper(model), input "input" (N, C, H, W) float32 = already-transformed
spectra (see SpectralTransform), output "logits" (N, K, H, W). N, H, W are dynamic (U-Net needs
H, W divisible by 8). Opset 17: supported by ONNX Runtime Web (WASM and WebGPU).

Variants
  fp32  - reference
  fp16  - weights and activations in float16, float32 inputs/outputs kept (keep_io_types)
  int8  - dynamic quantisation (weights int8, activations quantised at run time; ConvInteger)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from hsi.models import DenseWrapper

OPSET = 17


def export_onnx(
    model: torch.nn.Module, n_bands: int, path: Path, metadata: dict | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dense = DenseWrapper(model.cpu().eval()).eval()
    x = torch.randn(1, n_bands, 240, 320)
    torch.onnx.export(
        dense,
        (x,),
        str(path),
        opset_version=OPSET,
        dynamo=False,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}},
        do_constant_folding=True,
    )
    m = onnx.load(str(path))
    for k, v in (metadata or {}).items():
        m.metadata_props.add(key=k, value=v if isinstance(v, str) else json.dumps(v))
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return path


def to_fp16(src: Path, dst: Path) -> Path:
    from onnxconverter_common import float16

    m = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(m, str(dst))
    return dst


def to_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def session(
    path: Path, providers=("CPUExecutionProvider",), threads: int = 0
) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads  # 0 = ORT default (all physical cores)
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=list(providers))


def ort_logits(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    return sess.run(["logits"], {"input": x.astype(np.float32)})[0]


def parity(torch_logits: np.ndarray, onnx_logits: np.ndarray) -> dict:
    return {
        "max_abs_diff": float(np.abs(torch_logits - onnx_logits).max()),
        "argmax_agreement": float((torch_logits.argmax(1) == onnx_logits.argmax(1)).mean()),
    }


def _stats(ms: list[float]) -> dict:
    a = np.array(ms)
    med = float(np.median(a))
    return {
        "median_ms": med,
        "p90_ms": float(np.percentile(a, 90)),
        "fps": 1000.0 / med,
        "n": len(a),
    }


def bench_ort(sess, x: np.ndarray, warmup: int = 10, iters: int = 50) -> dict:
    feed = {"input": x.astype(np.float32)}
    for _ in range(warmup):
        sess.run(["logits"], feed)
    ms = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(["logits"], feed)
        ms.append((time.perf_counter() - t) * 1e3)
    return _stats(ms)


@torch.no_grad()
def bench_torch(model, x: np.ndarray, device: str, warmup: int = 10, iters: int = 50) -> dict:
    dense = DenseWrapper(model).to(device).eval()
    xt = torch.from_numpy(x).to(device)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        dense(xt)
    sync()
    ms = []
    for _ in range(iters):
        t = time.perf_counter()
        dense(xt)
        sync()
        ms.append((time.perf_counter() - t) * 1e3)
    return _stats(ms)


def bench_fn(fn, warmup: int = 5, iters: int = 30) -> dict:
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t) * 1e3)
    return _stats(ms)
