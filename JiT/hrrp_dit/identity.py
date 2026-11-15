"""Auxiliary identity loss for conditional HRRP diffusion.

The diffusion objective alone gives the source HRRP only a weak gradient: at
low noise the target itself already contains the answer, and at high noise the
noise-prediction target is dominated by ``x_t``.  On BUAA this lets the model
converge to an *azimuth-conditional but class-agnostic* generator: outputs from
different source classes become almost identical.

To keep the source identity, we add a small auxiliary cross-entropy term.  The
predicted clean HRRP ``x0_hat`` is normalised exactly like real profiles and
fed to a frozen 1-D VGG-16 classifier trained on real data; the target class is
the source/target class (they are equal by construction).  The term is applied
only at low-to-moderate noise levels, where ``x0_hat`` is meaningful.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import build_classifier
from .utils import scaled_to_hrrp


class IdentityLoss(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        signal_scale: float,
        weight: float = 0.1,
        t_max: int = 400,
        target_mode: str = "absolute",
        residual_scale: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        clf = build_classifier(payload.get("config", {}))
        clf.load_state_dict(payload["model"])
        clf.to(device).eval()
        for p in clf.parameters():
            p.requires_grad_(False)
        self.classifier = clf
        self.signal_scale = float(signal_scale)
        self.weight = float(weight)
        self.t_max = int(t_max)
        self.target_mode = str(target_mode)
        self.residual_scale = float(residual_scale)
        self.device = torch.device(device)

    def forward(
        self,
        x0_pred_scaled: torch.Tensor,
        batch: dict,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        with torch.no_grad():
            mask = (t <= self.t_max).to(x0_pred_scaled.dtype)
        if float(mask.sum()) == 0.0:
            z = x0_pred_scaled.sum() * 0.0
            return z, {"identity_ce": 0.0, "identity_frac": 0.0}

        if self.target_mode == "residual":
            src = batch["x_src_unit"].squeeze(1).to(x0_pred_scaled.dtype)
            x = src + x0_pred_scaled.squeeze(1) * self.residual_scale
            x = x.clamp_min(0.0)
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            x = x.unsqueeze(1)
        else:
            x = scaled_to_hrrp(x0_pred_scaled, self.signal_scale)
        logits = self.classifier(x)
        ce = F.cross_entropy(logits.float(), batch["y"], reduction="none")
        aux = (ce * mask).sum() / mask.sum().clamp_min(1.0)
        stats = {
            "identity_ce": float(aux.detach()),
            "identity_frac": float(mask.mean()),
        }
        return self.weight * aux, stats
