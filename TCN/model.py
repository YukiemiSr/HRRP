"""Original 144178-parameter BUAA TCN; original checkpoints load without key conversion."""

from __future__ import annotations
import torch
from torch import nn

if __package__:
    from .tcn import TemporalConvNet
else:
    from tcn import TemporalConvNet


class PositionHead(nn.Sequential):
    def __init__(self, channels: int, length: int, classes: int) -> None:
        super().__init__(
            nn.Conv1d(channels, 8, 1), nn.Flatten(), nn.Linear(8 * length, classes)
        )


class TCNClassifier(nn.Module):
    def __init__(self, input_size: int = 800, num_classes: int = 10) -> None:
        super().__init__()
        if input_size < 16 or num_classes < 2:
            raise ValueError("Expected at least 16 range bins and 2 classes.")
        self.input_size = input_size
        self.backbone = TemporalConvNet(1, [32] * 6, kernel_size=7, dropout=0.1)
        self.head = PositionHead(32, input_size, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (1, self.input_size):
            raise ValueError(
                f"Expected [B,1,{self.input_size}], got {tuple(values.shape)}"
            )
        return self.head(self.backbone(values))

    def description(self) -> dict:
        return {
            "name": "tcn",
            "input_size": self.input_size,
            "parameters": sum(p.numel() for p in self.parameters()),
            "conditioning": False,
            "position_grid_length": self.input_size,
            "normalization": "weight_norm",
        }
