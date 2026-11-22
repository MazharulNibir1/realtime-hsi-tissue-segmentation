from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PatchCNN(nn.Module):
    """Spatial-spectral patch classifier with an exact receptive field of `patch_size`.

    1x1 spectral reduction, then (patch_size - 1) / 2 unpadded 3x3 conv blocks, then a 1x1
    classifier. A (patch_size x patch_size) patch -> (1 x 1) output: the centre pixel's class.
    `forward_dense` reflect-pads the frame by (patch_size - 1) / 2, so one pass over a full
    frame gives the same result as classifying every pixel's patch separately (in eval mode).
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        patch_size: int = 11,
        spectral_channels: int = 32,
        width: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_size % 2 == 0 or patch_size < 3:
            raise ValueError("patch_size must be odd and >= 3")
        self.receptive_field = patch_size
        self.pad = (patch_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, spectral_channels, 1),
            nn.BatchNorm2d(spectral_channels),
            nn.ReLU(inplace=True),
        ]
        c = spectral_channels
        for _ in range(self.pad):
            layers += [nn.Conv2d(c, width, 3), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            c = width
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(c, n_classes, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, k, k) patches -> (N, K) centre-pixel logits."""
        y = self.net(x)
        return y[:, :, 0, 0] if y.shape[-2:] == (1, 1) else y

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, K, H, W)."""
        return self.net(F.pad(x, (self.pad,) * 4, mode="reflect"))
