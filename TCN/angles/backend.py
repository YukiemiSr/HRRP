"""Explicit cuDNN policy, applied after seeding and recorded in every run."""

from __future__ import annotations

import torch

MODES = ("strict", "fast")


def configure_backend(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(f"Choose cuDNN mode from {MODES}.")
    if mode == "fast" and torch.are_deterministic_algorithms_enabled():
        raise ValueError(
            "Global deterministic algorithms prevent the fast dilated convolution path."
        )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = mode == "strict"
    torch.backends.cudnn.benchmark = False
    return {
        "mode": mode,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "global_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
