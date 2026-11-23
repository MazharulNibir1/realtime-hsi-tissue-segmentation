#!/usr/bin/env python
"""T8: export demo samples (CC BY 4.0 HeiPorSPECTRAL, modified) to app/samples and web/public/samples.

Samples come from the TEST pigs only. Each processed 240x320 frame is mean-binned 2x2 again to
120x160 and L1 re-normalised (3.8 MB per sample in float16). Both demos read identical files.

    python scripts/make_samples.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from hsi.data.dataset import ProcessedStore
from hsi.data.heiporspectral import IGNORE
from hsi.data.normalize import SpectralTransform
from hsi.data.spectral import WAVELENGTHS_NM, bin_spatial, downsample_labels, l1_normalize
from hsi.export import ort_logits, session
from hsi.samples import CLASS_COLOURS, pseudo_rgb_bands, pseudo_rgb_params, write_sample

ROOT = Path(__file__).resolve().parents[1]
# Four organs the demo model handles well on the test pigs + kidney, its documented failure case.
ORGANS = ["liver", "spleen", "stomach", "colon", "kidney"]
DESTS = [ROOT / "app/samples", ROOT / "web/public/samples"]
CITATION = (
    "Studier-Fischer A., Seidlitz S., Sellner J., et al. HeiPorSPECTRAL - the Heidelberg Porcine "
    "HyperSPECTRAL Imaging Dataset of 20 Physiological Organs. Scientific Data 10, 414 (2023). "
    "doi:10.1038/s41597-023-02315-8"
)
CHANGES = (
    "Taken from the dataset's L1-normalised cubes (480x640x100); mean-binned 4x4 to 120x160 "
    "(2x2 in preprocessing, 2x2 here), L1 re-normalised per pixel, stored as float16. Labels: annotator 1 "
    "polygon mask, kept only where all source pixels are annotated."
)

LICENCE_MD = f"""# Sample data licence

These files are derived from **HeiPorSPECTRAL**, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Citation: {CITATION}

Changes: {CHANGES}

Research demo only - not for clinical use.
"""


def main() -> int:
    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    store = ProcessedStore(ROOT / data_cfg["processing"]["processed_dir"], "annotator1")
    test_pigs = data_cfg["split"]["test"]
    bench = json.loads((ROOT / "results/benchmark.json").read_text())
    spectra = json.loads((ROOT / "results/class_mean_spectra.json").read_text())

    # demo models (T7 choice) for the per-sample accuracy shown in the demos
    models = {}
    for bands, c in bench["demo_choice"].items():
        p = ROOT / c["path"]
        models[bands] = (
            session(p),
            SpectralTransform.from_dict(json.loads((p.parent / "transform.json").read_text())),
        )

    staging = ROOT / "exports/samples"
    if staging.exists():
        shutil.rmtree(staging)
    entries = []
    for organ in ORGANS:
        cands = sorted(
            (
                e
                for e in store.entries.values()
                if e["label"] == organ and e["subject"] in test_pigs
            ),
            key=lambda e: (e["situs"], e["subject"], e["timestamp"]),
        )
        e = cands[0]
        cube = l1_normalize(bin_spatial(np.asarray(store.cube(e["image_name"]), np.float32), 2))
        labels = downsample_labels(store.labels(e["image_name"]), 2, IGNORE)
        acc = {}
        for bands, (sess, tr) in models.items():
            pred = ort_logits(sess, tr(cube).transpose(2, 0, 1)[None].copy())[0].argmax(0)
            m = labels != IGNORE
            acc[bands] = float((pred[m] == labels[m]).mean())
        sid = f"{organ}_{e['subject']}"
        meta = write_sample(staging, sid, cube.astype(np.float16), labels, {
            "organ": organ,
            "label_idx": e["label_idx"],
            "source_image": e["image_name"],
            "subject": e["subject"],
            "split": "test",
            "situs": e["situs"], "angle": e["angle"], "repetition": e["repetition"],
            "wavelengths_nm": WAVELENGTHS_NM.round(3).tolist(),
            "pseudo_rgb": pseudo_rgb_params(cube, pseudo_rgb_bands(WAVELENGTHS_NM)),
            "annotated_pixels": int((labels != IGNORE).sum()),
            "demo_model_pixel_accuracy": acc,
            "normalisation": "cube is L1-normalised per pixel; model input = transform.json applied to it",
        })  # fmt: skip
        entries.append(meta)
        print(f"{sid}: {cube.shape} annotated={meta['annotated_pixels']} acc={acc}")

    index = {
        "samples": [{k: m[k] for k in ("id", "organ", "subject", "source_image", "demo_model_pixel_accuracy")}
                    for m in entries],  # fmt: skip
        "class_names": spectra["class_names"],
        "class_colours": CLASS_COLOURS,
        "wavelengths_nm": WAVELENGTHS_NM.round(3).tolist(),
        "class_mean_spectra": {
            s: {"band_centres_nm": spectra["band_centres_nm"][s],
                "mean": np.round(v["mean"], 6).tolist(), "std": np.round(v["std"], 6).tolist()}
            for s, v in spectra["settings"].items()
        },  # fmt: skip
        "class_mean_spectra_source": spectra["source"],
        "licence": "CC BY 4.0",
        "citation": CITATION,
        "changes": CHANGES,
        "disclaimer": "Research demo only - not for clinical use.",
    }
    (staging / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    (staging / "LICENSE.md").write_text(LICENCE_MD)

    for dest in DESTS:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(staging, dest)
    total = sum(f.stat().st_size for f in DESTS[0].iterdir())
    print(
        f"Wrote {len(entries)} samples ({total / 1e6:.1f} MB) to "
        + ", ".join(str(d.relative_to(ROOT)) for d in DESTS)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
