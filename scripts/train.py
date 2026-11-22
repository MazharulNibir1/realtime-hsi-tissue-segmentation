#!/usr/bin/env python
"""Train one model for one band setting and seed.

python scripts/train.py --model configs/models/unet.yaml --bands snapshot16 --seed 0
python scripts/train.py --model configs/models/pixel_mlp.yaml --set train.epochs=1 train.limit_images=8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from hsi.train import apply_overrides, train

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="model YAML (configs/models/*.yaml)")
    ap.add_argument("--data", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--bands", default="full", help="key under `bands:` in the data config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="override model config"
    )
    ap.add_argument("--force", action="store_true", help="retrain even if the run finished")
    args = ap.parse_args()

    model_cfg = apply_overrides(yaml.safe_load(Path(args.model).read_text()), args.set)
    data_cfg = yaml.safe_load(Path(args.data).read_text())
    if args.bands not in data_cfg["bands"]:
        ap.error(f"--bands must be one of {list(data_cfg['bands'])}")
    summary = train(model_cfg, data_cfg, args.bands, args.seed, Path(args.out), ROOT, args.force)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
