"""1-D VGG-16 HRRP classifier used to measure identity preservation.

The classifier is trained on *real* profiles only.  Generated profiles are
then fed to it; a high accuracy means the synthesised HRRP still carries the
correct target identity.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class VGG16Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (64, 128, 256, 512, 512),
        blocks: Sequence[int] = (2, 2, 3, 3, 3),
        in_len: int = 800,
    ):
        super().__init__()
        if len(channels) != len(blocks):
            raise ValueError("channels and blocks must have the same length")
        stages = []
        c_in = in_channels
        for c_out, n_conv in zip(channels, blocks):
            layers = []
            for _ in range(n_conv):
                layers += [
                    nn.Conv1d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(inplace=True),
                ]
                c_in = c_out
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.out_channels = int(channels[-1])
        self.out_len = int(in_len) // (2 ** len(channels))

    def forward(self, x: torch.Tensor, return_block: int | None = None) -> torch.Tensor:
        h = x
        for i, stage in enumerate(self.stages):
            h = stage(h)
            if return_block is not None and i == return_block:
                return h
        return h


class HRRPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_len: int = 800,
        channels: Sequence[int] = (64, 128, 256, 512, 512),
        blocks: Sequence[int] = (2, 2, 3, 3, 3),
        hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = VGG16Encoder(1, channels, blocks, in_len)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(int(channels[-1]), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(self.pool(h).flatten(1))


def build_classifier(cfg: dict) -> HRRPClassifier:
    c = cfg.get("classifier", {})
    m = cfg.get("model", {})
    return HRRPClassifier(
        num_classes=int(c.get("num_classes", 10)),
        in_len=int(cfg.get("data", {}).get("length", 800)),
        channels=tuple(c.get("channels", (64, 128, 256, 512, 512))),
        blocks=tuple(c.get("blocks", (2, 2, 3, 3, 3))),
        hidden=int(c.get("hidden", 256)),
        dropout=float(c.get("dropout", 0.3)),
    )
