"""Checkpoint loading and batched conditional sampling."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .diffusion import GaussianDiffusion
from .model import build_dit, count_parameters
from .utils import scaled_to_hrrp


@dataclass
class Generator:
    model: torch.nn.Module
    diffusion: GaussianDiffusion
    cfg: dict
    signal_scale: float
    target_mode: str = "absolute"
    residual_scale: float = 1.0
    step: int = 0
    device: torch.device = torch.device("cpu")

    @property
    def num_parameters(self) -> int:
        return count_parameters(self.model)


def load_generator(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> Generator:
    device = torch.device(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = build_dit(cfg).to(device)
    state = payload.get("ema") if use_ema and "ema" in payload else payload["model"]
    model.load_state_dict(state)
    model.eval()

    diffusion = GaussianDiffusion(cfg.get("diffusion", {})).to(device)
    diffusion.eval()

    extra = payload.get("extra", {})
    scale = float(extra.get("signal_scale", cfg.get("data", {}).get("signal_scale", 1.0)))
    if scale <= 0:
        raise ValueError(f"invalid signal scale {scale}")
    return Generator(
        model=model,
        diffusion=diffusion,
        cfg=cfg,
        signal_scale=scale,
        target_mode=str(extra.get("target_mode", cfg.get("data", {}).get("target_mode", "absolute"))),
        residual_scale=float(extra.get("residual_scale", cfg.get("data", {}).get("residual_scale", 1.0))),
        step=int(payload.get("step", 0)),
        device=device,
    )


@torch.no_grad()
def generate_profiles(
    gen: Generator,
    x_src: np.ndarray,
    a_src: np.ndarray,
    a_tgt: np.ndarray,
    batch_size: int = 256,
    steps: int = 50,
    cfg_scale: float = 1.0,
    sampler: str = "ddim",
    eta: float = 0.0,
    seed: int = 0,
    normalize: bool = True,
    residual_shrink: float = 1.0,
    progress: bool = True,
) -> np.ndarray:
    """Generate HRRPs conditioned on ``(x_src, a_src)`` at azimuth ``a_tgt``.

    Parameters
    ----------
    x_src : (N, L) unit-L2 measured HRRPs
    a_src, a_tgt : (N,) azimuths in degrees
    normalize : if True, map the network output back to a non-negative
        unit-L2 HRRP (the representation used by all metrics)
    """
    x_src = np.asarray(x_src, dtype=np.float32)
    a_src = np.asarray(a_src, dtype=np.float32)
    a_tgt = np.asarray(a_tgt, dtype=np.float32)
    if not (len(x_src) == len(a_src) == len(a_tgt)):
        raise ValueError("x_src, a_src, a_tgt must have the same length")
    gen.model.eval()
    gen.diffusion.eval()
    generator = torch.Generator(device=gen.device)
    generator.manual_seed(int(seed))

    out = []
    n = len(x_src)
    starts = range(0, n, int(batch_size))
    if progress:
        try:
            from tqdm import tqdm

            starts = tqdm(list(starts), desc="sampling", leave=False)
        except Exception:
            pass
    for i in starts:
        sl = slice(i, min(i + int(batch_size), n))
        src_unit = torch.from_numpy(x_src[sl]).to(gen.device)
        cond = (src_unit / gen.signal_scale).unsqueeze(1)
        asrc = torch.from_numpy(a_src[sl]).to(gen.device)
        atgt = torch.from_numpy(a_tgt[sl]).to(gen.device)
        x0 = gen.diffusion.sample(
            gen.model,
            cond,
            asrc,
            atgt,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            eta=eta,
            generator=generator,
        )
        if gen.target_mode == "residual":
            target = src_unit + float(residual_shrink) * x0.squeeze(1) * gen.residual_scale
            if normalize:
                target = target.clamp_min(0.0)
                target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            x0 = target.unsqueeze(1)
        elif normalize:
            x0 = scaled_to_hrrp(x0, gen.signal_scale)
        out.append(x0.squeeze(1).float().cpu().numpy())
    if not out:
        return np.zeros((0, x_src.shape[1]), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)


def generate_dataset(
    gen: Generator,
    data: Dict[str, object],
    batch_size: int = 256,
    steps: int = 50,
    cfg_scale: float = 1.0,
    sampler: str = "ddim",
    seed: int = 0,
    residual_shrink: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Generate for every entry of a dictionary produced by ``data.build_*``."""
    x_src = np.asarray(data["x_src"], dtype=np.float32)
    a_src = np.asarray(data["a_src"], dtype=np.float32)
    a_tgt = np.asarray(data["a_tgt"], dtype=np.float32)
    x_gen = generate_profiles(
        gen,
        x_src,
        a_src,
        a_tgt,
        batch_size=batch_size,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        seed=seed,
        residual_shrink=residual_shrink,
    )
    return dict(
        x_gen=x_gen,
        x_src=x_src,
        a_src=a_src,
        a_tgt=a_tgt,
        y=np.asarray(data["y"], dtype=np.int64),
    )
