"""Original 1D TCN with per-block, identity-initialized angle FiLM."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..model import TCNClassifier

CONDITIONS = ("none", "yaw", "yaw_pitch", "pitch")


class AngleTCN(TCNClassifier):
    def __init__(
        self, input_size: int = 800, num_classes: int = 10, conditioning: str = "yaw"
    ) -> None:
        if conditioning not in CONDITIONS:
            raise ValueError(f"Choose conditioning from {CONDITIONS}.")
        super().__init__(input_size, num_classes)
        self.conditioning = conditioning
        dimension = 4 if conditioning == "yaw_pitch" else 2
        self.films = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dimension, 32), nn.SiLU(), nn.Linear(32, 64))
                for _ in self.backbone.network
            ]
            if conditioning != "none"
            else []
        )
        for film in self.films:
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)

    def forward(
        self, values: torch.Tensor, angles: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.conditioning == "none":
            return super().forward(values)
        if values.ndim != 3 or values.shape[1:] != (1, self.input_size):
            raise ValueError(f"Expected HRRP [B,1,{self.input_size}].")
        columns = 2 if self.conditioning == "yaw_pitch" else 1
        if angles is None or angles.shape != (len(values), columns):
            raise ValueError(f"Expected degree-valued angles [B,{columns}].")
        if not torch.isfinite(angles).all():
            raise ValueError("Angle conditions must be finite.")
        radians = angles.to(device=values.device, dtype=values.dtype) * (math.pi / 180)
        # Each supplied angle contributes sin and cos. Joint order is yaw, pitch.
        condition = torch.stack((radians.sin(), radians.cos()), dim=-1).flatten(1)
        features = values
        for block, film in zip(self.backbone.network, self.films):
            features = block(features)
            gamma, beta = film(condition).chunk(2, dim=1)
            features = (1 + gamma.unsqueeze(-1)) * features + beta.unsqueeze(-1)
        return self.head(features)

    def description(self) -> dict:
        return {
            **super().description(),
            "conditioning": self.conditioning,
            "condition_features": {
                "none": [],
                "yaw": ["sin_yaw", "cos_yaw"],
                "pitch": ["sin_pitch", "cos_pitch"],
                "yaw_pitch": ["sin_yaw", "cos_yaw", "sin_pitch", "cos_pitch"],
            }[self.conditioning],
            "film_placement": "after each of six TemporalBlocks",
            "film_hidden_dim": 32,
            "film_initialization": "gamma=beta=0; identity modulation",
            "film_parameters": sum(p.numel() for p in self.films.parameters()),
            "inference_requires_yaw": self.conditioning in ("yaw", "yaw_pitch"),
            "inference_requires_pitch": self.conditioning in ("pitch", "yaw_pitch"),
        }
