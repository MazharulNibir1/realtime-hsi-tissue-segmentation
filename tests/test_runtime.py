import torch

import hsi
from hsi.utils import get_device, seed_everything


def test_version():
    assert hsi.__version__


def test_get_device_returns_cpu_or_mps():
    assert get_device().type in {"cpu", "mps"}
    assert get_device("cpu").type == "cpu"


def test_seed_is_reproducible():
    seed_everything(123)
    a = torch.rand(4)
    seed_everything(123)
    b = torch.rand(4)
    assert torch.equal(a, b)
