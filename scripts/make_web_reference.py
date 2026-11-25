#!/usr/bin/env python
"""Reference outputs for the browser parity test (web/tests/parity.test.ts).

For each sample in SAMPLES and each demo model in web/public/models, Python computes:
  - the transformed model input at fixed pixels (hsi.data.normalize.SpectralTransform),
  - the ONNX Runtime (CPU, Python) logits at those pixels,
  - the full argmax map and the accuracy on annotated pixels.
The TypeScript port + onnxruntime-web must reproduce them within tolerance.

    python scripts/make_web_reference.py   ->  web/tests/fixtures/reference.json
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

from hsi.data.normalize import SpectralTransform
from hsi.export import ort_logits, session
from hsi.samples import read_sample

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web/public"
OUT = ROOT / "web/tests/fixtures/reference.json"
SAMPLES = ["liver_P095", "kidney_P096"]
N_PIXELS = 64


def main() -> int:
    ref = {"samples": {}}
    for sid in SAMPLES:
        cube, labels, meta = read_sample(WEB / "samples", sid)
        h, w, _ = cube.shape
        cube32 = cube.astype(np.float32)
        rng = np.random.default_rng(0)
        pix = np.concatenate(
            [[0, w - 1, (h - 1) * w, h * w - 1], rng.choice(h * w, N_PIXELS - 4, replace=False)]
        )
        entry = {"pixels": pix.tolist(), "models": {}}
        for mdir in sorted((WEB / "models").iterdir()):
            tr = SpectralTransform.from_dict(json.loads((mdir / "transform.json").read_text()))
            x = tr(cube32)  # (H, W, B)
            logits = ort_logits(
                session(mdir / "model.onnx"), np.ascontiguousarray(x.transpose(2, 0, 1)[None])
            )[0]
            pred = logits.argmax(0).astype(np.uint8)
            m = labels != 255
            flat_x = x.reshape(h * w, -1)
            flat_l = logits.reshape(logits.shape[0], -1)
            entry["models"][mdir.name] = {
                "input_at_pixels": flat_x[pix].round(7).tolist(),
                "logits_at_pixels": flat_l[:, pix].T.round(6).tolist(),
                "argmax_b64": base64.b64encode(pred.tobytes()).decode(),
                "annotated_accuracy": float((pred[m] == labels[m]).mean()),
            }
            print(f"{sid}/{mdir.name}: acc={entry['models'][mdir.name]['annotated_accuracy']:.4f}")
        ref["samples"][sid] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ref) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
