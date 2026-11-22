#!/usr/bin/env python
"""Raw HeiPorSPECTRAL subset -> data/processed (binned float16 cubes + uint8 label maps).

Idempotent: images whose outputs exist are skipped (only index.json is rebuilt).

    python scripts/preprocess_data.py [--limit N] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from hsi.data.heiporspectral import load_class_names, process_image, raw_paths
from hsi.data.spectral import WAVELENGTHS_NM
from hsi.data.split import make_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_root = ROOT / cfg["paths"]["raw_dir"] / cfg["dataset"]["name"]
    out_dir = ROOT / cfg["processing"]["processed_dir"]
    factor = cfg["processing"]["bin_factor"]
    annotators = cfg["selection"]["annotators"]
    manifest = json.loads((ROOT / cfg["paths"]["manifest"]).read_text())
    class_names = load_class_names(raw_root / "data/dataset_settings.json")
    label_idx = {n: i for i, n in enumerate(class_names)}

    entries = manifest["images"][: args.limit] if args.limit else manifest["images"]
    missing = [
        e["image_name"]
        for e in entries
        if not all(
            p.exists()
            for p in [
                raw_paths(raw_root, e, annotators)[0],
                *raw_paths(raw_root, e, annotators)[1].values(),
            ]
        )
    ]
    if missing:
        print(
            f"ERROR: {len(missing)} images missing raw files (run `make data`), e.g. {missing[:3]}"
        )
        return 2

    workers = args.workers or cfg["processing"]["workers"]
    metas = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [
            ex.submit(
                process_image, e, raw_root, out_dir, label_idx[e["label"]], factor, annotators
            )
            for e in entries
        ]
        for f in tqdm(as_completed(futs), total=len(futs), desc="preprocess", unit="img"):
            metas.append(f.result())
    metas.sort(key=lambda m: m["image_name"])

    split = make_split({m["subject"] for m in metas}, cfg["split"]["val"], cfg["split"]["test"])
    index = {
        "dataset": cfg["dataset"]["name"],
        "licence": manifest["licence"],
        "bin_factor": factor,
        "wavelengths_nm": WAVELENGTHS_NM.round(3).tolist(),
        "class_names": class_names,
        "ignore_index": 255,
        "images": metas,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")

    for part, subs in split.items():
        imgs = [m for m in metas if m["subject"] in subs]
        px = sum(m[f"labelled_px_{cfg['processing']['label_source']}"] for m in imgs)
        n_cls = len({m["label"] for m in imgs})
        print(f"{part:5s}: pigs={subs} images={len(imgs)} classes={n_cls} labelled_px={px:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
