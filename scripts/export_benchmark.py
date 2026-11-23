#!/usr/bin/env python
"""T7: ONNX export (fp32/fp16/int8), parity, test F1 per variant, latency benchmarks, demo choice.

For every (model, bands) the seed with the best *validation* macro F1 is exported to
exports/<model>_<bands>/ (git-ignored) together with transform.json.

Latency: batch 1, full frame (C x 240 x 320; also 480 x 640 for fp32), real transformed test frame,
10 warm-up runs excluded, median of 50 timed runs. "Model" latency excludes preprocessing;
preprocessing (band projection + L1 + standardise, NumPy) is timed separately.

Demo model choice (one per band setting): speed first, because both demos run on CPU-class
backends (the local Gradio app on ONNX Runtime CPU, and ONNX Runtime Web WASM in the browser):
  1. candidates: fp32 models, file < 10 MB
  2. lowest ONNX Runtime CPU latency at 240x320
  3. precision: fp16/int8 replaces fp32 only if it is smaller AND agrees with fp32 on >= 99.5%
     of test-frame pixel predictions AND is >= 10% faster on ONNX Runtime CPU (beyond timing noise)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from hsi.data.dataset import ProcessedStore
from hsi.data.heiporspectral import IGNORE
from hsi.data.split import make_split
from hsi.evaluate import TestSet
from hsi.export import (
    bench_fn,
    bench_ort,
    bench_torch,
    export_onnx,
    ort_logits,
    parity,
    session,
    to_fp16,
    to_int8,
)
from hsi.metrics import summarize
from hsi.models import DenseWrapper
from hsi.train import load_run

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("fp32", "fp16", "int8")
MODEL_LABEL = {"pixel_mlp": "Pixel MLP", "patch_cnn": "Patch CNN", "unet": "U-Net"}


def best_seed_runs(runs: Path) -> dict[tuple[str, str], Path]:
    best: dict[tuple[str, str], tuple[float, Path]] = {}
    for f in runs.glob("*/summary.json"):
        s = json.loads(f.read_text())
        key = (s["model"], s["bands"])
        if key not in best or s["best_val_macro_f1"] > best[key][0]:
            best[key] = (s["best_val_macro_f1"], f.parent)
    return {k: v[1] for k, v in sorted(best.items())}


def test_macro_f1_onnx(sess, transform, test: TestSet, k: int) -> tuple[float, np.ndarray]:
    cm = np.zeros((k, k), np.int64)
    preds = []
    for cube, lab in zip(test.cubes, test.labels, strict=True):
        x = transform(cube).transpose(2, 0, 1)[None]
        p = ort_logits(sess, np.ascontiguousarray(x))[0].argmax(0).astype(np.uint8)
        preds.append(p)
        m = lab != IGNORE
        cm += np.bincount(lab[m].astype(np.int64) * k + p[m], minlength=k * k).reshape(k, k)
    return summarize(cm)["macro_f1"], np.stack(preds)


def machine() -> dict:
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    ).stdout.strip()
    import onnxruntime as ort

    return {"cpu": chip or platform.processor(), "os": platform.platform(), "torch": torch.__version__,
            "onnxruntime": ort.__version__, "python": platform.python_version()}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--exports", default=str(ROOT / "exports"))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--report-only", action="store_true", help="re-render benchmark.md from benchmark.json"
    )
    args = ap.parse_args()
    if args.report_only:  # re-apply the selection rule and re-render from saved measurements
        rep = json.loads((ROOT / "results/benchmark.json").read_text())
        rep["demo_choice"] = choose_demo(rep["rows"])
        (ROOT / "results/benchmark.json").write_text(json.dumps(rep, indent=1) + "\n")
        write_markdown(rep, ROOT / "results/benchmark.md")
        return 0
    exports = Path(args.exports)

    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    store = ProcessedStore(
        ROOT / data_cfg["processing"]["processed_dir"], data_cfg["processing"]["label_source"]
    )
    split = make_split(store.subjects, data_cfg["split"]["val"], data_cfg["split"]["test"])
    k = len(store.class_names)
    test = TestSet(store, store.images(split["test"]))
    t6 = json.loads((ROOT / "results/test_metrics.json").read_text())["aggregate"]
    mps = torch.backends.mps.is_available()

    rows = []
    for (name, bands), run_dir in tqdm(best_seed_runs(Path(args.runs)).items(), desc="models"):
        model, transform, cfg = load_run(run_dir, "cpu")
        out = exports / f"{name}_{bands}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "transform.json").write_text((run_dir / "transform.json").read_text())
        meta = {"model": name, "bands": bands, "seed": str(cfg["seed"]), "run": run_dir.name,
                "class_names": cfg["class_names"], "input": "transformed spectra (see transform.json)"}  # fmt: skip
        paths = {"fp32": export_onnx(model, transform.n_bands, out / "model_fp32.onnx", meta)}
        paths["fp16"] = to_fp16(paths["fp32"], out / "model_fp16.onnx")
        paths["int8"] = to_int8(paths["fp32"], out / "model_int8.onnx")

        frame = np.ascontiguousarray(
            transform(test.cubes[0]).transpose(2, 0, 1)[None]
        )  # 1xCx240x320
        frame_big = np.ascontiguousarray(np.repeat(np.repeat(frame, 2, 2), 2, 3))  # 1xCx480x640
        with torch.no_grad():
            ref = DenseWrapper(model)(torch.from_numpy(frame)).numpy()
        pre = bench_fn(lambda tr=transform: tr(test.cubes[0]))

        fp32_preds = None
        for v in VARIANTS:
            sess = session(paths[v])
            par = parity(ref, ort_logits(sess, frame))
            f1, preds = test_macro_f1_onnx(sess, transform, test, k)
            if v == "fp32":
                fp32_preds = preds
            row = {
                "model": name, "bands": bands, "variant": v, "seed": cfg["seed"],
                "params": int(sum(p.numel() for p in model.parameters())),
                "size_mb": paths[v].stat().st_size / 1e6,
                "test_macro_f1_this_seed": f1,
                "test_macro_f1_3seeds": t6[f"{name}/{bands}"]["macro_f1"],
                "val_macro_f1_3seeds": t6[f"{name}/{bands}"]["val_macro_f1"],
                "parity_vs_torch_frame0": par,
                "test_pixel_agreement_vs_fp32": float((preds == fp32_preds).mean()),
                "ort_cpu_240x320": bench_ort(sess, frame, iters=args.iters),
                "preprocess_numpy_240x320": pre,
                "path": str(paths[v].relative_to(ROOT)),
            }  # fmt: skip
            if v == "fp32":
                row["ort_cpu_480x640"] = bench_ort(sess, frame_big, iters=max(10, args.iters // 2))
                row["torch_cpu_240x320"] = bench_torch(model, frame, "cpu", iters=args.iters)
                if mps:
                    row["torch_mps_240x320"] = bench_torch(model, frame, "mps", iters=args.iters)
                    row["torch_mps_480x640"] = bench_torch(
                        model, frame_big, "mps", iters=args.iters
                    )
                    model.cpu()
            rows.append(row)
            print(f"{name}/{bands}/{v}: size={row['size_mb']:.2f}MB F1={f1:.4f} "
                  f"ort_cpu={row['ort_cpu_240x320']['median_ms']:.1f}ms max|d|={par['max_abs_diff']:.2e}", flush=True)  # fmt: skip

    choice = choose_demo(rows)
    report = {"machine": machine(), "test_pigs": split["test"], "rows": rows, "demo_choice": choice}
    (ROOT / "results/benchmark.json").write_text(json.dumps(report, indent=1) + "\n")
    write_markdown(report, ROOT / "results/benchmark.md")
    print((ROOT / "results/benchmark.md").read_text())
    return 0


def choose_demo(rows: list[dict]) -> dict:
    """Fastest fp32 model on ORT CPU per band setting (file < 10 MB), then the precision rule."""
    choice = {}
    for bands in sorted({r["bands"] for r in rows}):
        cands = [
            r for r in rows if r["bands"] == bands and r["variant"] == "fp32" and r["size_mb"] < 10
        ]
        if not cands:
            continue
        pick = min(cands, key=lambda r: r["ort_cpu_240x320"]["median_ms"])
        alt = [r for r in rows if r["bands"] == bands and r["model"] == pick["model"] and r["variant"] != "fp32"
               and r["size_mb"] < pick["size_mb"] and r["test_pixel_agreement_vs_fp32"] >= 0.995
               and r["ort_cpu_240x320"]["median_ms"] < 0.9 * pick["ort_cpu_240x320"]["median_ms"]]  # fmt: skip
        if alt:
            pick = min(alt, key=lambda r: r["size_mb"])
        choice[bands] = {"model": pick["model"], "variant": pick["variant"], "path": pick["path"]}
    return choice


def write_markdown(rep: dict, path: Path) -> None:
    def ms(d):
        return f"{d['median_ms']:.1f} ms ({d['fps']:.0f} fps)"

    m = rep["machine"]
    L = ["# Export and latency benchmark", "",
         f"Machine: {m['cpu']}, {m['os']}; PyTorch {m['torch']}, ONNX Runtime {m['onnxruntime']}. "
         "Batch 1, full frame, 10 warm-up runs excluded, median of timed runs. Model latency excludes "
         "preprocessing (shown separately). F1 columns: test pigs " + ", ".join(rep["test_pigs"]) + ".", "",
         "## Precision variants (ONNX Runtime CPU, 240 × 320)", "",
         "| Model | Bands | Variant | Params | File | Test macro F1 (exported seed) | Pixel agreement vs fp32 | Max abs logit diff vs PyTorch | ORT CPU |",
         "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rep["rows"]:
        L.append(f"| {MODEL_LABEL[r['model']]} | {r['bands']} | {r['variant']} | {r['params'] / 1e3:.0f}k | "
                 f"{r['size_mb']:.2f} MB | {r['test_macro_f1_this_seed']:.3f} | {r['test_pixel_agreement_vs_fp32']:.4f} | "
                 f"{r['parity_vs_torch_frame0']['max_abs_diff']:.1e} | {ms(r['ort_cpu_240x320'])} |")  # fmt: skip
    L += ["", "## fp32 latency by runtime", "",
          "| Model | Bands | Test macro F1 (3 seeds) | Preprocess (NumPy) | PyTorch CPU 240×320 | ORT CPU 240×320 | ORT CPU 480×640 | PyTorch MPS 240×320 | PyTorch MPS 480×640 |",
          "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rep["rows"]:
        if r["variant"] != "fp32":
            continue
        f = r["test_macro_f1_3seeds"]
        L.append(f"| {MODEL_LABEL[r['model']]} | {r['bands']} | {f['mean']:.3f} ± {f['std']:.3f} | "
                 f"{ms(r['preprocess_numpy_240x320'])} | {ms(r['torch_cpu_240x320'])} | {ms(r['ort_cpu_240x320'])} | "
                 f"{ms(r['ort_cpu_480x640'])} | {ms(r['torch_mps_240x320']) if 'torch_mps_240x320' in r else 'n/a'} | "
                 f"{ms(r['torch_mps_480x640']) if 'torch_mps_480x640' in r else 'n/a'} |")  # fmt: skip
    L += ["", "## Demo models", "", "Rule: speed first, because both demos run on CPU-class backends (local Gradio app on "
          "ONNX Runtime CPU, browser demo on ONNX Runtime Web WASM). Among fp32 models with file < 10 MB, the lowest ONNX Runtime CPU latency "
          "at 240 × 320 wins; a fp16/int8 variant replaces fp32 only if it is smaller, agrees with fp32 on ≥ 99.5% "
          "of test pixels and is ≥ 10% faster (beyond timing noise).", ""]  # fmt: skip
    for b, c in rep["demo_choice"].items():
        L.append(f"- **{b}:** {MODEL_LABEL[c['model']]} ({c['variant']}) → `{c['path']}`")
    notes = path.parent / "benchmark_notes.md"
    if notes.exists():
        L += ["", notes.read_text().rstrip()]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
