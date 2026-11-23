#!/usr/bin/env python
"""T6: evaluate every finished run on the test pigs; write results/ (tables, figures, JSON).

    python scripts/evaluate.py            # all runs/*/summary.json
Outputs
  results/results.md                generated tables + results/analysis.md (hand-written) appended
  results/test_metrics.json         per-run and aggregated metrics (incl. confusion matrices)
  results/class_mean_spectra.json   per-class mean/std spectra (training pigs) - used by the demos
  results/figures/*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from hsi.data.dataset import ProcessedStore
from hsi.data.spectral import WAVELENGTHS_NM, band_centres, band_matrix
from hsi.data.split import make_split
from hsi.evaluate import (
    TestSet,
    aggregate,
    class_mean_spectra,
    evaluate_run,
    save_json,
    top_confusions,
)
from hsi.plots import confusion_figure, per_class_figure, pretty, spectra_figure
from hsi.utils import get_device

ROOT = Path(__file__).resolve().parents[1]
MODEL_LABEL = {"pixel_mlp": "Pixel MLP", "patch_cnn": "Patch CNN (11×11)", "unet": "U-Net"}
BAND_LABEL = {"full": "full (100 bands)", "snapshot16": "16-band"}


def pm(d: dict, nd: int = 3) -> str:
    return f"{d['mean']:.{nd}f} ± {d['std']:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    store = ProcessedStore(
        ROOT / data_cfg["processing"]["processed_dir"], data_cfg["processing"]["label_source"]
    )
    split = make_split(store.subjects, data_cfg["split"]["val"], data_cfg["split"]["test"])
    names = store.class_names
    k = len(names)
    device = get_device()

    run_dirs = sorted(p.parent for p in Path(args.runs).glob("*/summary.json"))
    if not run_dirs:
        print("No finished runs found")
        return 1
    print(f"Loading {len(store.images(split['test']))} test images into RAM ...")
    test = TestSet(store, store.images(split["test"]))
    per_run = []
    for rd in tqdm(run_dirs, desc="evaluate", unit="run"):
        r = evaluate_run(rd, test, k, device)
        s = json.loads((rd / "summary.json").read_text())
        r.update({"val_macro_f1": s["best_val_macro_f1"], "params": s["params"], "best_epoch": s["best_epoch"],
                  "train_time_s": s["train_time_s"]})  # fmt: skip
        per_run.append(r)
    agg = aggregate(per_run)
    for key, a in agg.items():
        rs = [r for r in per_run if f"{r['model']}/{r['bands']}" == key]
        a["val_macro_f1"] = {"mean": float(np.mean([r["val_macro_f1"] for r in rs])),
                             "std": float(np.std([r["val_macro_f1"] for r in rs], ddof=1))}  # fmt: skip
        a["params"] = rs[0]["params"]
        a["train_time_min"] = float(np.mean([r["train_time_s"] for r in rs]) / 60)
    save_json(out / "test_metrics.json", {"test_pigs": split["test"], "class_names": names, "runs": per_run,
                                          "aggregate": agg})  # fmt: skip

    # ---- class mean spectra (training pigs) - also consumed by both demos
    W = {
        s: band_matrix(c["method"], c.get("n_bands", 16), c.get("fwhm_nm", 20.0))
        for s, c in data_cfg["bands"].items()
    }
    spectra = class_mean_spectra(store, store.images(split["train"]), W)
    spectra["band_centres_nm"] = {s: band_centres(w).round(2).tolist() for s, w in W.items()}
    spectra["source"] = f"annotated pixels of training pigs {split['train']}"
    save_json(out / "class_mean_spectra.json", spectra)

    # ---- figures
    spectra_figure(
        spectra,
        WAVELENGTHS_NM,
        np.array(spectra["band_centres_nm"]["snapshot16"]),
        figs / "mean_spectra.png",
    )
    for a in agg.values():
        confusion_figure(np.array(a["confusion_matrix_sum"]), names,
                         f"{MODEL_LABEL[a['model']]}, {BAND_LABEL[a['bands']]}: test pigs, 3 seeds pooled "
                         f"(row-normalised)", figs / f"confusion_{a['model']}_{a['bands']}.png")  # fmt: skip
    best_model = max({a["model"] for a in agg.values()},
                     key=lambda m: agg[f"{m}/full"]["macro_f1"]["mean"] + agg[f"{m}/snapshot16"]["macro_f1"]["mean"])  # fmt: skip
    af, a16 = agg[f"{best_model}/full"], agg[f"{best_model}/snapshot16"]
    per_class_figure(names, af["per_class_f1_mean"], a16["per_class_f1_mean"], af["per_class_f1_std"],
                     a16["per_class_f1_std"], f"{MODEL_LABEL[best_model]}: per-class test F1",
                     figs / f"per_class_f1_{best_model}.png")  # fmt: skip

    # ---- results.md (generated)
    L = ["# Results", "",
         f"Generated by `scripts/evaluate.py`. Test pigs {', '.join(split['test'])} ({len(test.names)} images); "
         "metrics on annotated pixels only; mean ± sd over 3 seeds (0, 1, 2). Checkpoint = best validation "
         "macro F1 (val pigs " + ", ".join(split["val"]) + "). Pixel Dice equals pixel F1 under this protocol, "
         "so it is reported once.", "",
         "## Main table", "",
         "| Model | Bands | Params | Test macro F1 | Test pixel acc. | Image-level acc. | Val macro F1 | Train time |",
         "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for a in agg.values():
        L.append(f"| {MODEL_LABEL[a['model']]} | {BAND_LABEL[a['bands']]} | {a['params'] / 1e3:.0f}k | "
                 f"**{pm(a['macro_f1'])}** | {pm(a['accuracy'])} | {pm(a['image_accuracy'])} | "
                 f"{pm(a['val_macro_f1'])} | {a['train_time_min']:.1f} min |")  # fmt: skip
    L += ["", "Image-level accuracy: majority predicted class inside the annotated polygon equals the organ.", "",
          "## Per-pig test macro F1", "", "| Model | Bands | " + " | ".join(split["test"]) + " |",
          "|---|---|" + "---|" * len(split["test"])]  # fmt: skip
    for a in agg.values():
        L.append(f"| {MODEL_LABEL[a['model']]} | {BAND_LABEL[a['bands']]} | "
                 + " | ".join(pm(a["per_pig_macro_f1"][p]) for p in split["test"]) + " |")  # fmt: skip
    L += ["", "## Per-class test F1 (= pixel Dice), mean over seeds", "",
          "| Class | " + " | ".join(f"{MODEL_LABEL[a['model']]} {a['bands']}" for a in agg.values()) + " |",
          "|---|" + "---|" * len(agg)]  # fmt: skip
    for i, n in enumerate(names):
        L.append(
            f"| {pretty(n)} | "
            + " | ".join(f"{a['per_class_f1_mean'][i]:.2f}" for a in agg.values())
            + " |"
        )
    L += [
        "",
        "## Most frequent confusions (test, 3 seeds pooled; fraction of the true class's pixels)",
        "",
    ]
    for a in agg.values():
        tc = top_confusions(np.array(a["confusion_matrix_sum"]), names, 5)
        L.append(f"- **{MODEL_LABEL[a['model']]}, {BAND_LABEL[a['bands']]}:** "
                 + "; ".join(f"{pretty(t)} → {pretty(p)} {v:.2f}" for t, p, v in tc))  # fmt: skip
    L += ["", "## Figures", "",
          "![Class mean spectra](figures/mean_spectra.png)", "",
          f"![Per-class F1](figures/per_class_f1_{best_model}.png)", ""]  # fmt: skip
    for a in agg.values():
        L.append(f"- [Confusion matrix: {MODEL_LABEL[a['model']]}, {BAND_LABEL[a['bands']]}]"
                 f"(figures/confusion_{a['model']}_{a['bands']}.png)")  # fmt: skip
    analysis = out / "analysis.md"
    if analysis.exists():
        L += ["", analysis.read_text().rstrip()]
    (out / "results.md").write_text("\n".join(L) + "\n")
    print((out / "results.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
