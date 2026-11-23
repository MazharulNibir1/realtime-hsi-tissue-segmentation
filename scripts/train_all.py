#!/usr/bin/env python
"""T5: every model x band setting x seed, sequentially. Resumable (finished runs are skipped).

python scripts/train_all.py [--models pixel_mlp patch_cnn unet] [--bands full snapshot16] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

from hsi.train import train

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["pixel_mlp", "patch_cnn", "unet"])
    ap.add_argument("--bands", nargs="+", default=["full", "snapshot16"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "runs"))
    args = ap.parse_args()

    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    jobs = [(m, b, s) for m in args.models for b in args.bands for s in args.seeds]
    failed = []
    t0 = time.time()
    for k, (m, b, s) in enumerate(jobs, 1):
        print(
            f"\n=== [{k}/{len(jobs)}] {m} {b} seed={s} (elapsed {(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )
        model_cfg = yaml.safe_load((ROOT / f"configs/models/{m}.yaml").read_text())
        try:
            summary = train(model_cfg, data_cfg, b, s, Path(args.out), ROOT)
            print(json.dumps(summary), flush=True)
        except Exception:  # keep going; report at the end
            traceback.print_exc()
            failed.append((m, b, s))
    print(f"\nFinished {len(jobs) - len(failed)}/{len(jobs)} in {(time.time() - t0) / 60:.1f} min")
    for f in failed:
        print("FAILED:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
