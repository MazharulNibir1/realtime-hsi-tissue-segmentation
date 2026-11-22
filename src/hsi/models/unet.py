from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Small U-Net. Bands are input channels; a 1x1 stem first mixes the spectrum.

    H and W must be divisible by 2**depth (240x320 works up to depth 4).
    Upsampling uses nearest + conv (exports cleanly to ONNX / ONNX Runtime Web).
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        base: int = 16,
        depth: int = 3,
        stem_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.depth = depth
        self.receptive_field = None
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        chs = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        c = stem_channels
        for ch in chs[:-1]:
            self.down.append(_block(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(_block(c, chs[-1]), nn.Dropout2d(dropout))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = chs[-1]
        for ch in reversed(chs[:-1]):
            self.up.append(
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, ch, 1))
            )
            self.dec.append(_block(2 * ch, ch))
            c = ch
        self.head = nn.Conv2d(c, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
