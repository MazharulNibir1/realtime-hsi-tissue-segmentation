import json

import numpy as np
import pytest
import torch

from hsi.data.dataset import (
    ImageDataset,
    PatchDataset,
    PixelDataset,
    ProcessedStore,
    extract_patch,
    fit_transform_on,
)
from hsi.data.normalize import transform_from_config

H, W = 24, 32


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(0)
    images = []
    for k, (subj, cls) in enumerate([("P1", 0), ("P1", 1), ("P2", 2)]):
        name = f"{subj}#t{k}"
        cube = rng.random((H, W, 100)).astype(np.float32)
        cube = (cube / cube.sum(-1, keepdims=True)).astype(np.float16)
        lab = np.full((H, W), 255, np.uint8)
        lab[5:15, 8:20] = cls
        (tmp_path / "cubes").mkdir(exist_ok=True)
        (tmp_path / "labels/annotator1").mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / "cubes" / f"{name}.npy", cube)
        np.save(tmp_path / "labels/annotator1" / f"{name}.npy", lab)
        images.append({"image_name": name, "subject": subj, "label": f"c{cls}", "label_idx": cls})
    (tmp_path / "index.json").write_text(
        json.dumps({"class_names": ["c0", "c1", "c2"], "images": images})
    )
    return ProcessedStore(tmp_path)


@pytest.mark.parametrize(
    "setting,n", [({"method": "identity"}, 100), ({"method": "box", "n_bands": 16}, 16)]
)
def test_dataset_shapes(store, setting, n):
    names = store.images(["P1", "P2"])
    t = fit_transform_on(
        store, names, transform_from_config(setting, "l1_standardize"), n_per_image=50
    )

    px = PixelDataset(store, names, t, n_per_image=40)
    x, y = px[0]
    assert (
        len(px) == 120 and x.shape == (n,) and x.dtype == torch.float32 and y.dtype == torch.int64
    )

    pt = PatchDataset(store, names, t, patch_size=11, n_per_image=10, augment=True)
    x, y = pt[0]
    assert len(pt) == 30 and x.shape == (n, 11, 11)

    im = ImageDataset(store, names, t, augment=True)
    x, y = im[0]
    assert x.shape == (n, H, W) and y.shape == (H, W) and y.dtype == torch.int64
    assert set(np.unique(y.numpy())) <= {0, 1, 2, 255}


def test_subject_filter(store):
    assert store.images(["P2"]) == ["P2#t2"]
    assert store.subjects == ["P1", "P2"]


def test_extract_patch_borders():
    cube = np.arange(5 * 6 * 2, dtype=np.float32).reshape(5, 6, 2)
    for r, c in [(0, 0), (4, 5), (2, 3), (0, 5)]:
        p = extract_patch(cube, r, c, 5)
        assert p.shape == (5, 5, 2)
        np.testing.assert_array_equal(p[2, 2], cube[r, c])


def test_image_crops_contain_labels(store):
    from hsi.data.normalize import transform_from_config

    t = transform_from_config({"method": "box", "n_bands": 16}, "l1")
    ds = ImageDataset(store, store.images(["P1"]), t, augment=True, crop_size=8, crops_per_image=3)
    assert len(ds) == 6
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (16, 8, 8) and y.shape == (8, 8)
        assert (y != 255).any()  # every crop contains a labelled pixel
