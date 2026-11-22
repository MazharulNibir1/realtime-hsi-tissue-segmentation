"""Access to processed data + PyTorch datasets for the three model families.

PixelDataset  - (n_bands,) spectra of sampled labelled pixels          -> spectral MLP / 1D-CNN
PatchDataset  - (n_bands, k, k) patches centred on labelled pixels      -> spatial-spectral CNN
ImageDataset  - (n_bands, H, W) full frames + (H, W) labels (255 = ignore) -> U-Net
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from hsi.data.heiporspectral import IGNORE
from hsi.data.normalize import SpectralTransform


class ProcessedStore:
    def __init__(self, processed_dir: str | Path, label_source: str = "annotator1"):
        self.root = Path(processed_dir)
        idx = json.loads((self.root / "index.json").read_text())
        self.class_names: list[str] = idx["class_names"]
        self.entries: dict[str, dict] = {e["image_name"]: e for e in idx["images"]}
        self.label_source = label_source

    @property
    def subjects(self) -> list[str]:
        return sorted({e["subject"] for e in self.entries.values()})

    def images(self, subjects: list[str]) -> list[str]:
        s = set(subjects)
        return sorted(n for n, e in self.entries.items() if e["subject"] in s)

    def cube(self, name: str) -> np.ndarray:
        """(H, W, 100) float16, memory-mapped."""
        return np.load(self.root / "cubes" / f"{name}.npy", mmap_mode="r")

    def labels(self, name: str) -> np.ndarray:
        return np.load(self.root / "labels" / self.label_source / f"{name}.npy")


def sample_labelled_pixels(
    labels: np.ndarray, n: int | None, rng: np.random.Generator
) -> np.ndarray:
    """(k, 2) row/col coordinates of up to n labelled pixels (all if n is None)."""
    coords = np.argwhere(labels != IGNORE)
    if n is not None and len(coords) > n:
        coords = coords[rng.choice(len(coords), n, replace=False)]
    return coords


def extract_patch(cube: np.ndarray, r: int, c: int, size: int) -> np.ndarray:
    """(size, size, C) patch centred on (r, c); reflect-padded at image borders."""
    h, w = cube.shape[:2]
    half = size // 2
    r0, r1, c0, c1 = r - half, r + half + 1, c - half, c + half + 1
    patch = np.asarray(cube[max(r0, 0) : min(r1, h), max(c0, 0) : min(c1, w)])
    pad = ((max(-r0, 0), max(r1 - h, 0)), (max(-c0, 0), max(c1 - w, 0)), (0, 0))
    if any(p for pp in pad for p in pp):
        patch = np.pad(patch, pad, mode="reflect")
    return patch


def fit_transform_on(
    store: ProcessedStore,
    names: list[str],
    transform: SpectralTransform,
    n_per_image: int = 2000,
    seed: int = 0,
) -> SpectralTransform:
    """Fit normalisation statistics on random pixels (labelled or not) of training images."""
    if not transform.needs_fit:
        return transform
    rng = np.random.default_rng(seed)
    samples = []
    for n in names:
        cube = store.cube(n)
        h, w = cube.shape[:2]
        rr, cc = rng.integers(0, h, n_per_image), rng.integers(0, w, n_per_image)
        samples.append(np.asarray(cube[rr, cc], dtype=np.float32))
    return transform.fit(np.concatenate(samples))


def _rand_aug() -> tuple[bool, bool, int]:
    """Random flips / 90-degree rotation from torch's RNG (seeded per DataLoader worker)."""
    r = torch.randint(0, 2, (2,))
    return bool(r[0]), bool(r[1]), int(torch.randint(0, 4, (1,)))


class PixelDataset(Dataset):
    """Pre-sampled labelled pixels held in memory (transformed once)."""

    def __init__(
        self, store, names, transform: SpectralTransform, n_per_image: int | None = 1000, seed=0
    ):
        rng = np.random.default_rng(seed)
        xs, ys = [], []
        for n in names:
            lab = store.labels(n)
            coords = sample_labelled_pixels(lab, n_per_image, rng)
            if len(coords) == 0:
                continue
            spectra = np.asarray(store.cube(n)[coords[:, 0], coords[:, 1]], dtype=np.float32)
            xs.append(transform(spectra))
            ys.append(lab[coords[:, 0], coords[:, 1]])
        self.x = torch.from_numpy(np.concatenate(xs))
        self.y = torch.from_numpy(np.concatenate(ys).astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class PatchDataset(Dataset):
    """Patches extracted on the fly from memory-mapped cubes."""

    def __init__(
        self, store, names, transform: SpectralTransform, patch_size: int = 11,
        n_per_image: int | None = 500, seed=0, augment: bool = False,
    ):  # fmt: skip
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        rng = np.random.default_rng(seed)
        self.store, self.names, self.transform = store, names, transform
        self.size, self.augment = patch_size, augment
        items = []
        for i, n in enumerate(names):
            lab = store.labels(n)
            coords = sample_labelled_pixels(lab, n_per_image, rng)
            items.append(
                np.column_stack([np.full(len(coords), i), coords, lab[coords[:, 0], coords[:, 1]]])
            )
        self.items = np.concatenate(items).astype(np.int64)
        self._mmaps: dict[int, np.ndarray] = {}

    def _cube(self, i: int) -> np.ndarray:
        if i not in self._mmaps:  # mmap handles are cheap; data is paged in lazily
            self._mmaps[i] = self.store.cube(self.names[i])
        return self._mmaps[i]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, r, c, y = self.items[k]
        p = self.transform(extract_patch(self._cube(int(i)), int(r), int(c), self.size))
        if self.augment:
            flip_h, flip_v, k = _rand_aug()
            if flip_h:
                p = p[:, ::-1]
            if flip_v:
                p = p[::-1]
            p = np.rot90(p, k)
        x = torch.from_numpy(np.ascontiguousarray(p.transpose(2, 0, 1)))
        return x, torch.tensor(y)


class ImageDataset(Dataset):
    """Full frames (or random crops) for segmentation. Unlabelled pixels carry IGNORE (255).

    crop_size: if set, each item is a crop_size x crop_size window that contains a randomly
    chosen labelled pixel (cropped before the spectral transform, so it is also cheaper).
    crops_per_image: dataset length multiplier when cropping.
    """

    def __init__(
        self, store, names, transform: SpectralTransform, augment: bool = False,
        crop_size: int | None = None, crops_per_image: int = 1,
    ):  # fmt: skip
        self.store, self.names, self.transform, self.augment = store, names, transform, augment
        self.crop_size = crop_size
        self.k = crops_per_image if crop_size else 1

    def __len__(self):
        return len(self.names) * self.k

    def _crop_window(self, lab: np.ndarray) -> tuple[slice, slice]:
        h, w = lab.shape
        cs = self.crop_size
        coords = np.argwhere(lab != IGNORE)
        if len(coords):
            r, c = coords[int(torch.randint(len(coords), (1,)))]
        else:
            r, c = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
        r0 = int(np.clip(r - int(torch.randint(cs, (1,))), 0, h - cs))
        c0 = int(np.clip(c - int(torch.randint(cs, (1,))), 0, w - cs))
        return slice(r0, r0 + cs), slice(c0, c0 + cs)

    def __getitem__(self, i):
        n = self.names[i // self.k]
        y = self.store.labels(n)
        cube = self.store.cube(n)
        if self.crop_size:
            rs, cs = self._crop_window(y)
            cube, y = cube[rs, cs], y[rs, cs]
        x = self.transform(np.asarray(cube))
        y = y.astype(np.int64)
        if self.augment:
            flip_h, flip_v, k = _rand_aug()
            if flip_h:
                x, y = x[:, ::-1], y[:, ::-1]
            if flip_v:
                x, y = x[::-1], y[::-1]
            if self.crop_size:  # square crops: 90-degree rotations are shape-preserving
                x, y = np.rot90(x, k), np.rot90(y, k)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        return x, torch.from_numpy(np.ascontiguousarray(y))
