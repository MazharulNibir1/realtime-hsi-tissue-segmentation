#!/usr/bin/env python
"""Copy the T7 demo models (results/benchmark.json -> demo_choice) into both demos.

  app/models/<bands>/{model.onnx, transform.json, model.json}
  web/public/models/<bands>/{model.onnx, transform.json, model.json}

model.json carries provenance + the measured numbers shown in the UIs.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTS = [ROOT / "app/models", ROOT / "web/public/models"]
LABEL = {"full": "Full spectrum (100 bands)", "snapshot16": "Simulated 16-band snapshot sensor"}


def main() -> int:
    bench = json.loads((ROOT / "results/benchmark.json").read_text())
    for dest in DESTS:
        if dest.exists():
            shutil.rmtree(dest)
    for bands, c in bench["demo_choice"].items():
        src = ROOT / c["path"]
        row = next(r for r in bench["rows"] if r["path"] == c["path"])
        transform = json.loads((src.parent / "transform.json").read_text())
        info = {
            "bands": bands,
            "label": LABEL.get(bands, bands),
            "model": c["model"],
            "variant": c["variant"],
            "n_bands": transform["n_bands"],
            "seed": row["seed"],
            "params": row["params"],
            "size_bytes": src.stat().st_size,
            "test_macro_f1_3seeds": row["test_macro_f1_3seeds"],
            "ort_cpu_ms_240x320_m3pro": round(row["ort_cpu_240x320"]["median_ms"], 2),
            "input": "float32 (1, n_bands, H, W) = transform.json applied to an L1-normalised 100-band cube",
            "output": "float32 (1, 20, H, W) logits",
        }
        for dest in DESTS:
            d = dest / bands
            d.mkdir(parents=True)
            shutil.copy2(src, d / "model.onnx")
            (d / "transform.json").write_text(json.dumps(transform) + "\n")
            (d / "model.json").write_text(json.dumps(info, indent=1) + "\n")
        print(f"{bands}: {c['model']} {c['variant']} ({src.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
