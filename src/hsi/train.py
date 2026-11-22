"""Training loop shared by all models. Logging: CSV + JSON per run (no external service).

Run directory runs/<model>_<bands>_s<seed>/ contains
  config.yaml      fully resolved config (model + data + bands + seed)
  transform.json   input transform (band matrix, normalisation stats) - needed for inference
  metrics.csv      one row per epoch
  best.pt          weights with the best validation macro F1
  summary.json     best epoch/metrics, params, timings, device

Validation is identical for every model: dense full-frame inference on all validation images,
metrics on every annotated pixel. Only the training sampling differs per model.
"""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from hsi.data.dataset import (
    ImageDataset,
    PatchDataset,
    PixelDataset,
    ProcessedStore,
    fit_transform_on,
)
from hsi.data.heiporspectral import IGNORE
from hsi.data.normalize import SpectralTransform, transform_from_config
from hsi.data.split import make_split
from hsi.metrics import confusion_matrix, summarize
from hsi.models import DenseWrapper, build_model, count_params
from hsi.utils import get_device, seed_everything


def set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """`key.sub=value` strings; values parsed as YAML (numbers, lists, bools)."""
    cfg = copy.deepcopy(cfg)
    for o in overrides:
        k, v = o.split("=", 1)
        set_by_path(cfg, k, yaml.safe_load(v))
    return cfg


def class_weights(counts: np.ndarray, scheme: str) -> torch.Tensor:
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1)
    if scheme == "none":
        w = np.ones_like(counts)
    elif scheme == "inverse":
        w = 1 / counts
    elif scheme == "sqrt_inverse":
        w = 1 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class_weighting {scheme!r}")
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def lr_at(
    progress: float, base_lr: float, warmup_frac: float = 0.05, final_frac: float = 0.01
) -> float:
    """Linear warm-up then cosine decay to final_frac * base_lr; progress in [0, 1]."""
    warm = min(1.0, (progress + 1e-3) / warmup_frac) if warmup_frac > 0 else 1.0
    cos = 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
    return float(base_lr * warm * (final_frac + (1 - final_frac) * cos))


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)  # noqa: NPY002


def build_eval_cache(
    store, names, transform, batch_size=4
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Transform validation frames once and keep them in RAM (float16 inputs, uint8 labels)."""
    dl = DataLoader(ImageDataset(store, names, transform), batch_size=batch_size, num_workers=2)
    return [(x.half(), y.to(torch.uint8)) for x, y in dl]


@torch.no_grad()
def evaluate_dense(model, batches, device, n_classes) -> dict:
    """Full-frame inference; confusion matrix + mean CE over annotated pixels."""
    model.eval()
    dense = DenseWrapper(model).to(device)
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    loss_sum, n = 0.0, 0
    for x, y in batches:
        x, y = x.to(device).float(), y.to(device).long()
        logits = dense(x)
        mask = y != IGNORE
        if mask.any():
            loss_sum += F.cross_entropy(logits, y, ignore_index=IGNORE, reduction="sum").item()
            n += int(mask.sum())
        cm += confusion_matrix(logits.argmax(1).cpu(), y.cpu(), n_classes)
    out = summarize(cm.numpy())
    out["loss"] = loss_sum / max(n, 1)
    out["confusion_matrix"] = cm.tolist()
    return out


def _make_train_ds(kind, store, names, transform, dcfg, seed):
    if kind == "pixel":
        return PixelDataset(store, names, transform, dcfg.get("n_per_image"), seed)
    if kind == "patch":
        return PatchDataset(
            store, names, transform, dcfg["patch_size"], dcfg.get("n_per_image"), seed, augment=True
        )
    if kind == "image":
        return ImageDataset(
            store, names, transform, augment=True,
            crop_size=dcfg.get("crop_size"), crops_per_image=dcfg.get("crops_per_image", 1),
        )  # fmt: skip
    raise ValueError(f"Unknown dataset kind {kind!r}")


def _label_counts(kind, ds, store, names, n_classes) -> np.ndarray:
    if kind == "pixel":
        return np.bincount(ds.y.numpy(), minlength=n_classes)
    if kind == "patch":
        return np.bincount(ds.items[:, 3], minlength=n_classes)
    counts = np.zeros(n_classes)
    for n in names:
        e = store.entries[n]
        counts[e["label_idx"]] += e[f"labelled_px_{store.label_source}"]
    return counts


def train(
    model_cfg: dict, data_cfg: dict, bands: str, seed: int, out_root: Path, root: Path, force=False
) -> dict:
    name = model_cfg["name"]
    run_dir = out_root / f"{name}_{bands}_s{seed}"
    if (run_dir / "summary.json").exists() and not force:
        print(f"[skip] {run_dir} already finished (use --force to retrain)")
        return json.loads((run_dir / "summary.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = get_device(model_cfg["train"].get("device", "auto"))
    pcfg = data_cfg["processing"]
    store = ProcessedStore(root / pcfg["processed_dir"], pcfg["label_source"])
    split = make_split(store.subjects, data_cfg["split"]["val"], data_cfg["split"]["test"])
    train_names, val_names = store.images(split["train"]), store.images(split["val"])
    if lim := model_cfg["train"].get("limit_images"):  # smoke tests only
        train_names, val_names = train_names[:lim], val_names[: max(1, lim // 2)]
    n_classes = len(store.class_names)

    transform = transform_from_config(data_cfg["bands"][bands], data_cfg["normalization"])
    transform = fit_transform_on(store, train_names, transform, seed=seed)
    (run_dir / "transform.json").write_text(json.dumps(transform.to_dict()))

    resolved = {
        "model": model_cfg,
        "bands": bands,
        "band_config": data_cfg["bands"][bands],
        "normalization": data_cfg["normalization"],
        "label_source": pcfg["label_source"],
        "split": split,
        "seed": seed,
        "class_names": store.class_names,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    model = build_model(name, transform.n_bands, n_classes, **model_cfg.get("model", {})).to(device)
    n_params = count_params(model)
    tcfg, dcfg = model_cfg["train"], model_cfg["data"]
    kind = dcfg["kind"]
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    epochs = tcfg["epochs"]

    print(
        f"{run_dir.name}: {n_params:,} params | device={device} | train={len(train_names)} val={len(val_names)} imgs"
    )
    fields = ["epoch", "lr", "train_loss", "val_loss", "val_macro_f1", "val_accuracy", "epoch_s"]
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        csv.writer(f).writerow(fields)

    val_cache = build_eval_cache(store, val_names, transform)
    best, best_epoch, weights = -1.0, -1, None
    t_start = time.time()
    ds = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if ds is None or dcfg.get("resample_each_epoch", False):
            ds = _make_train_ds(kind, store, train_names, transform, dcfg, seed * 1000 + epoch)
            if weights is None:
                counts = _label_counts(kind, ds, store, train_names, n_classes)
                weights = class_weights(counts, tcfg.get("class_weighting", "none")).to(device)
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        nw = tcfg.get("num_workers", 0)
        dl = DataLoader(
            ds, batch_size=tcfg["batch_size"], shuffle=True, generator=g, num_workers=nw,
            worker_init_fn=_seed_worker, persistent_workers=False, drop_last=len(ds) > tcfg["batch_size"],
        )  # fmt: skip
        model.train()
        tot, cnt = 0.0, 0
        n_batches = len(dl)
        for i, (x, y) in enumerate(dl):
            progress = (epoch - 1 + i / n_batches) / epochs
            for group in opt.param_groups:
                group["lr"] = lr_at(progress, tcfg["lr"], tcfg.get("warmup_frac", 0.05))
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = F.cross_entropy(model(x), y, weight=weights, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            cnt += len(x)
        lr = opt.param_groups[0]["lr"]

        if epoch % tcfg.get("val_every", 1) and epoch != epochs:
            continue
        val = evaluate_dense(model, val_cache, device, n_classes)
        row = [epoch, f"{lr:.2e}", f"{tot / max(cnt, 1):.4f}", f"{val['loss']:.4f}",
               f"{val['macro_f1']:.4f}", f"{val['accuracy']:.4f}", f"{time.time() - t0:.1f}"]  # fmt: skip
        with (run_dir / "metrics.csv").open("a", newline="") as f:
            csv.writer(f).writerow(row)
        print("  " + " ".join(f"{k}={v}" for k, v in zip(fields, row, strict=True)), flush=True)
        if val["macro_f1"] > best:
            best, best_epoch = val["macro_f1"], epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
            (run_dir / "val_best.json").write_text(json.dumps(val))

    summary = {
        "run": run_dir.name,
        "model": name,
        "bands": bands,
        "seed": seed,
        "params": n_params,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best,
        "train_time_s": round(time.time() - t_start, 1),
        "device": str(device),
        "torch": torch.__version__,
        "git_commit": _git_commit(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_run(run_dir: Path, device: str | torch.device = "cpu"):
    """(model in eval mode, SpectralTransform, resolved config) for a finished run."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    transform = SpectralTransform.from_dict(json.loads((run_dir / "transform.json").read_text()))
    m = cfg["model"]
    model = build_model(m["name"], transform.n_bands, len(cfg["class_names"]), **m.get("model", {}))
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), transform, cfg
