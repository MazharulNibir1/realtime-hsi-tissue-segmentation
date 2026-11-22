from __future__ import annotations

import torch
from torch import nn


class PixelMLP(nn.Module):
    """Per-pixel spectral MLP implemented with 1x1 convolutions.

    Training input: (N, C) spectra or (N, C, 1, 1). Inference: (N, C, H, W) -> (N, K, H, W).
    """

    receptive_field = 1

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hidden: tuple[int, ...] = (128, 128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for h in hidden:
            layers += [nn.Conv2d(c, h, 1), nn.BatchNorm2d(h), nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout2d(dropout))
            c = h
        layers.append(nn.Conv2d(c, n_classes, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x[:, :, None, None]
        y = self.net(x)
        return y[:, :, 0, 0] if squeeze else y

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
