"""Three model families, all mapping (N, C, H, W) -> (N, K, H, W) logits at inference.

PixelMLP   - per-pixel spectral classifier built from 1x1 convs (no spatial context)
PatchCNN   - spatial-spectral; valid 3x3 convs give a fixed receptive field (= patch size).
             Trained on k x k patches (-> 1x1 output); at inference the frame is reflect-padded
             so one forward pass equals classifying every pixel's k x k patch.
UNet       - small encoder-decoder segmentation network, bands as input channels
"""

from __future__ import annotations

import torch
from torch import nn

from hsi.models.patch_cnn import PatchCNN
from hsi.models.pixel_mlp import PixelMLP
from hsi.models.unet import UNet

REGISTRY: dict[str, type[nn.Module]] = {"pixel_mlp": PixelMLP, "patch_cnn": PatchCNN, "unet": UNet}


def build_model(name: str, in_channels: int, n_classes: int, **kwargs) -> nn.Module:
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](in_channels=in_channels, n_classes=n_classes, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class DenseWrapper(nn.Module):
    """Full-frame inference module (used for evaluation and ONNX export)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_dense(x)


__all__ = ["PixelMLP", "PatchCNN", "UNet", "build_model", "count_params", "DenseWrapper"]
