"""Static figures for results.md / README (matplotlib, light theme).

Colours (validated: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface):
full spectrum = blue #2a78d6, 16-band = orange #eb6834. Confusion matrices use a single-hue
blue sequential ramp. 20 classes are never shown as 20 hues: spectra use small multiples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FULL = "#2a78d6"
SNAP = "#eb6834"
BLUES = LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def pretty(name: str) -> str:
    return name.replace("kidney_with_Gerotas_fascia", "kidney+Gerota").replace("_", " ")


def confusion_figure(cm: np.ndarray, names: list[str], title: str, path: Path) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    r = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    k = len(names)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(r, cmap=BLUES, vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if r[i, j] >= 0.05:  # selective labels: diagonal + notable confusions
                txt = f"{r[i, j]:.2f}"
                txt = "1" if txt == "1.00" else txt.lstrip("0")
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if r[i, j] > 0.55 else INK)  # fmt: skip
    labels = [pretty(n) for n in names]
    ax.set_xticks(range(k), labels, rotation=60, ha="right")
    ax.set_yticks(range(k), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (annotated pixels)")
    ax.set_title(title, loc="left", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("fraction of true-class pixels", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def spectra_figure(
    spectra: dict, wavelengths: np.ndarray, centres16: np.ndarray, path: Path
) -> None:
    names = spectra["class_names"]
    full = spectra["settings"]["full"]
    s16 = spectra["settings"]["snapshot16"]
    mean, std = np.array(full["mean"]), np.array(full["std"])
    m16 = np.array(s16["mean"]) * (
        16 / 100
    )  # rescale: L1 over 16 vs 100 bands -> same per-band scale
    fig, axes = plt.subplots(4, 5, figsize=(12, 8.4), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        ax.fill_between(
            wavelengths, mean[i] - std[i], mean[i] + std[i], color=FULL, alpha=0.18, lw=0
        )
        ax.plot(
            wavelengths, mean[i], color=FULL, lw=1.6, label="full spectrum (100 bands), mean ± 1 sd"
        )
        ax.plot(centres16, m16[i], "o", color=SNAP, ms=3.5, mec=SURFACE, mew=0.6,
                label="simulated 16-band sensor, mean (rescaled)")  # fmt: skip
        ax.set_title(pretty(names[i]), loc="left", fontsize=9)
        ax.grid(True, axis="y")
        ax.tick_params(length=0)
    for ax in axes[-1]:
        ax.set_xlabel("wavelength (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("L1-normalised")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=2, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("Class mean spectra, annotated pixels of the 7 training pigs", x=0.01, y=0.965,
                 ha="left", fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_figure(names, f1_full, f1_16, sd_full, sd_16, title: str, path: Path) -> None:
    f1_full, f1_16 = np.array(f1_full), np.array(f1_16)
    order = np.argsort(f1_full)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    for yi, i in zip(y, order, strict=True):
        ax.plot([f1_full[i], f1_16[i]], [yi, yi], color=AXIS, lw=1, zorder=1)
    ax.errorbar(f1_full[order], y, xerr=np.array(sd_full)[order], fmt="o", color=FULL, ms=6,
                mec=SURFACE, mew=1, elinewidth=1, capsize=0, label="full spectrum", zorder=3)  # fmt: skip
    ax.errorbar(f1_16[order], y, xerr=np.array(sd_16)[order], fmt="o", color=SNAP, ms=6,
                mec=SURFACE, mew=1, elinewidth=1, capsize=0, label="16-band", zorder=3)  # fmt: skip
    ax.set_yticks(y, [pretty(names[i]) for i in order])
    ax.set_xlabel("test F1 (= pixel Dice), mean ± sd over 3 seeds")
    ax.set_xlim(min(0.0, np.nanmin([f1_full, f1_16]) - 0.05), 1.0)
    ax.grid(True, axis="x")
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
