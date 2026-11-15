"""Gaussian diffusion with classifier-free guidance for HRRP DiT.

The forward/reverse process follows the DDPM formulation used by DiT
(ADM hyper-parameters): a 1000-step linear variance schedule from 1e-4 to
2e-2 and epsilon prediction.  Sampling is implemented both as DDPM (250 or
1000 ancestral steps, as in the paper) and as deterministic DDIM (50 steps by
default), which is much faster for 1-D signals.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(
    schedule: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    cosine_s: float = 0.008,
) -> np.ndarray:
    if schedule == "linear":
        return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
    if schedule == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps, dtype=np.float64)
        ac = np.cos(((x / timesteps) + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1.0 - ac[1:] / ac[:-1]
        return np.clip(betas, 1e-8, 0.999)
    if schedule == "scaled_linear":
        return np.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=np.float64) ** 2
    raise ValueError(f"unknown beta schedule {schedule!r}")


class GaussianDiffusion(nn.Module):
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.timesteps = int(cfg.get("timesteps", 1000))
        self.formulation = str(cfg.get("formulation", "ddpm")).lower()
        if self.formulation not in ("ddpm", "jit"):
            raise ValueError("formulation must be 'ddpm' or 'jit'")
        self.prediction = str(cfg.get("prediction", "eps")).lower()
        if self.formulation == "jit":
            if self.prediction != "x":
                raise ValueError("formulation='jit' requires prediction='x'")
        elif self.prediction not in ("eps", "v"):
            raise ValueError("prediction must be 'eps' or 'v'")
        self.clip_denoised = cfg.get("clip_denoised", 24.0)
        self.min_snr_gamma = float(cfg.get("min_snr_gamma", 0.0))
        # JiT (https://arxiv.org/abs/2511.13720): z_t = t*x + (1-t)*eps with
        # t in [0, 1] (t=1 clean).  The network predicts x and the loss is the
        # v-loss ||(x_pred - z)/(1-t) - (x - z)/(1-t)||^2.
        self.noise_scale = float(cfg.get("noise_scale", 1.0))
        self.t_dist = str(cfg.get("t_dist", "logit_normal")).lower()
        self.t_mu = float(cfg.get("t_mu", -0.8))
        self.t_sigma = float(cfg.get("t_sigma", 0.8))
        self.denom_clip = float(cfg.get("denom_clip", 0.05))
        self.time_scale = float(cfg.get("time_scale", 1000.0))

        betas = make_beta_schedule(
            str(cfg.get("schedule", "linear")),
            self.timesteps,
            float(cfg.get("beta_start", 1e-4)),
            float(cfg.get("beta_end", 2e-2)),
            float(cfg.get("cosine_s", 0.008)),
        )
        alphas = 1.0 - betas
        ac = np.cumprod(alphas)
        ac_prev = np.append(1.0, ac[:-1])

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer("alphas_cumprod", torch.tensor(ac, dtype=torch.float32))
        self.register_buffer("alphas_cumprod_prev", torch.tensor(ac_prev, dtype=torch.float32))
        self.register_buffer("sqrt_alphas_cumprod", torch.tensor(np.sqrt(ac), dtype=torch.float32))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.tensor(np.sqrt(1.0 - ac), dtype=torch.float32),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.tensor(np.sqrt(1.0 / ac), dtype=torch.float32),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.tensor(np.sqrt(1.0 / ac - 1.0), dtype=torch.float32),
        )
        posterior_var = betas * (1.0 - ac_prev) / (1.0 - ac)
        self.register_buffer("posterior_variance", torch.tensor(posterior_var, dtype=torch.float32))
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.tensor(np.log(np.clip(posterior_var, 1e-20, None)), dtype=torch.float32),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.tensor(betas * np.sqrt(ac_prev) / (1.0 - ac), dtype=torch.float32),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.tensor((1.0 - ac_prev) * np.sqrt(alphas) / (1.0 - ac), dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    def _extract(self, arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        out = arr.to(t.device)[t]
        while out.dim() < ndim:
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x0.dim()) * x0
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.dim()) * noise
        )

    # ------------------------------------------------------------------
    def _training_target(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if self.prediction == "eps":
            return noise
        ab = self._extract(self.alphas_cumprod, t, x0.dim())
        return torch.sqrt(ab) * noise - torch.sqrt(1.0 - ab) * x0

    def sample_t(
        self,
        b: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Continuous time for the JiT formulation, in (0, 1)."""
        if self.t_dist == "logit_normal":
            s = torch.randn(b, device=device, generator=generator) * self.t_sigma + self.t_mu
            t = torch.sigmoid(s)
        elif self.t_dist == "uniform":
            t = torch.rand(b, device=device, generator=generator)
        else:
            raise ValueError(f"unknown t_dist {self.t_dist!r}")
        return t.clamp_(1e-4, 1.0 - 1e-4)

    def loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        cond_drop_prob: float = 0.0,
        generator: Optional[torch.Generator] = None,
        aux_loss: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x0 = batch["x"]
        b = x0.shape[0]
        cond_drop = None
        if cond_drop_prob > 0:
            cond_drop = torch.rand(b, device=x0.device, generator=generator) < cond_drop_prob

        if self.formulation == "jit":
            # z = t*x + (1-t)*eps; the network predicts x; v-loss.
            t = self.sample_t(b, x0.device, generator)
            tv = t.view(-1, 1, 1)
            noise = torch.randn(
                x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
            ) * self.noise_scale
            z = tv * x0 + (1.0 - tv) * noise
            pred_x = model(
                z, t * self.time_scale, batch["x_cond"], batch["a_src"], batch["a_tgt"], cond_drop
            )
            inv = 1.0 / torch.clamp(1.0 - t, min=self.denom_clip).view(-1, 1, 1)
            # Equivalent to ||x_pred - x||^2 / (1-t)^2 (JiT v-loss).
            v = (x0.float() - z.float()) * inv.float()
            v_pred = (pred_x.float() - z.float()) * inv.float()
            loss = F.mse_loss(v_pred, v, reduction="none").mean(dim=(1, 2)).mean()
            with torch.no_grad():
                x0_mse = F.mse_loss(pred_x.float(), x0.float()).detach()
            stats = {
                "loss": float(loss.detach()),
                "x0_mse": float(x0_mse),
                "t_mean": float(t.float().mean()),
            }
            if aux_loss is not None:
                t_int = (t * self.timesteps).long()
                aux, aux_stats = aux_loss(pred_x, batch, t_int)
                loss = loss + aux
                stats.update(aux_stats)
                stats["loss_total"] = float(loss.detach())
            return loss, stats

        t = torch.randint(0, self.timesteps, (b,), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t, batch["x_cond"], batch["a_src"], batch["a_tgt"], cond_drop)
        target = self._training_target(x0, noise, t)
        loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2))
        if self.min_snr_gamma > 0:
            ab = self._extract(self.alphas_cumprod, t, 1).flatten()
            snr = ab / (1.0 - ab)
            weight = torch.clamp(snr.new_tensor(self.min_snr_gamma) / snr, max=1.0)
            loss = loss * weight
        loss = loss.mean()
        stats = {"loss": float(loss.detach()), "t_mean": float(t.float().mean())}
        if aux_loss is not None:
            x0_pred, _ = self._predict_x0_eps(xt, t, pred)
            aux, aux_stats = aux_loss(x0_pred, batch, t)
            loss = loss + aux
            stats.update(aux_stats)
            stats["loss_total"] = float(loss.detach())
        return loss, stats

    # ------------------------------------------------------------------
    def _predict_x0_eps(
        self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ab = self._extract(self.alphas_cumprod, t, xt.dim())
        if self.prediction == "eps":
            x0 = (xt - torch.sqrt(1.0 - ab) * model_out) / torch.sqrt(ab)
            eps = model_out
        else:
            x0 = torch.sqrt(ab) * xt - torch.sqrt(1.0 - ab) * model_out
            eps = torch.sqrt(1.0 - ab) * xt + torch.sqrt(ab) * model_out
        if self.clip_denoised is not None:
            c = float(self.clip_denoised)
            x0 = x0.clamp(-c, c)
            eps = (xt - torch.sqrt(ab) * x0) / torch.sqrt(1.0 - ab).clamp_min(1e-12)
        return x0, eps

    def _model_eps(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        x_cond: torch.Tensor,
        a_src: torch.Tensor,
        a_tgt: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        if cfg_scale == 1.0:
            return model(xt, t, x_cond, a_src, a_tgt, None)
        # One batched forward pass for the conditional and unconditional branches.
        b = xt.shape[0]
        xt2 = torch.cat([xt, xt], dim=0)
        cond2 = torch.cat([x_cond, x_cond], dim=0)
        src2 = torch.cat([a_src, a_src], dim=0)
        tgt2 = torch.cat([a_tgt, a_tgt], dim=0)
        t2 = torch.cat([t, t], dim=0)
        drop = torch.cat(
            [
                torch.zeros(b, dtype=torch.bool, device=xt.device),
                torch.ones(b, dtype=torch.bool, device=xt.device),
            ]
        )
        out = model(xt2, t2, cond2, src2, tgt2, drop)
        eps_cond, eps_uncond = out.chunk(2, dim=0)
        return eps_uncond + float(cfg_scale) * (eps_cond - eps_uncond)

    def _model_x(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        x_cond: torch.Tensor,
        a_src: torch.Tensor,
        a_tgt: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Model output for the JiT formulation (already x, not eps)."""
        if cfg_scale == 1.0:
            return model(xt, t, x_cond, a_src, a_tgt, None)
        b = xt.shape[0]
        xt2 = torch.cat([xt, xt], dim=0)
        cond2 = torch.cat([x_cond, x_cond], dim=0)
        src2 = torch.cat([a_src, a_src], dim=0)
        tgt2 = torch.cat([a_tgt, a_tgt], dim=0)
        t2 = torch.cat([t, t], dim=0)
        drop = torch.cat(
            [
                torch.zeros(b, dtype=torch.bool, device=xt.device),
                torch.ones(b, dtype=torch.bool, device=xt.device),
            ]
        )
        out = model(xt2, t2, cond2, src2, tgt2, drop)
        x_cond_out, x_uncond_out = out.chunk(2, dim=0)
        return x_uncond_out + float(cfg_scale) * (x_cond_out - x_uncond_out)

    # ------------------------------------------------------------------
    def ddim_timesteps(self, steps: int) -> np.ndarray:
        steps = max(1, min(int(steps), self.timesteps))
        ts = np.linspace(0, self.timesteps - 1, steps, dtype=np.float64)
        return np.unique(np.round(ts).astype(np.int64))[::-1]

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        x_cond: torch.Tensor,
        a_src: torch.Tensor,
        a_tgt: torch.Tensor,
        steps: int = 50,
        cfg_scale: float = 1.0,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Deterministic DDIM sampling; returns ``x0`` in scaled units."""
        model.eval()
        b = x_cond.shape[0]
        shape = (b, 1, x_cond.shape[-1])
        xt = torch.randn(shape, device=x_cond.device, dtype=x_cond.dtype, generator=generator)
        timesteps = self.ddim_timesteps(steps)
        for i, t_np in enumerate(timesteps):
            t = torch.full((b,), int(t_np), device=x_cond.device, dtype=torch.long)
            eps = self._model_eps(model, xt, t, x_cond, a_src, a_tgt, cfg_scale)
            x0, eps = self._predict_x0_eps(xt, t, eps)

            prev_np = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
            ab_t = self.alphas_cumprod[int(t_np)]
            ab_prev = self.alphas_cumprod[prev_np] if prev_np >= 0 else torch.tensor(1.0, device=xt.device)
            sigma = 0.0
            if eta > 0 and prev_np >= 0:
                sigma = (
                    float(eta)
                    * torch.sqrt((1 - ab_prev) / (1 - ab_t))
                    * torch.sqrt(1 - ab_t / ab_prev)
                )
            c = torch.sqrt(torch.clamp(1 - ab_prev - sigma**2, min=0.0))
            xt = torch.sqrt(ab_prev) * x0 + c * eps
            if eta > 0 and prev_np >= 0:
                xt = xt + sigma * torch.randn(xt.shape, device=xt.device, dtype=xt.dtype, generator=generator)
        return xt

    @torch.no_grad()
    def ddpm_sample(
        self,
        model: nn.Module,
        x_cond: torch.Tensor,
        a_src: torch.Tensor,
        a_tgt: torch.Tensor,
        steps: int = 250,
        cfg_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Ancestral DDPM sampling with ``steps`` evenly spaced reverse steps."""
        model.eval()
        b = x_cond.shape[0]
        xt = torch.randn((b, 1, x_cond.shape[-1]), device=x_cond.device, dtype=x_cond.dtype, generator=generator)
        # Descending subset of the full 1000-step chain, always including 0.
        ts = np.unique(np.round(np.linspace(0, self.timesteps - 1, steps)).astype(np.int64))[::-1]
        for i, t_np in enumerate(ts):
            t = torch.full((b,), int(t_np), device=x_cond.device, dtype=torch.long)
            eps = self._model_eps(model, xt, t, x_cond, a_src, a_tgt, cfg_scale)
            x0, _ = self._predict_x0_eps(xt, t, eps)
            prev_np = int(ts[i + 1]) if i + 1 < len(ts) else -1
            if prev_np < 0:
                xt = x0
                continue
            # Exact posterior of the respaced sub-chain between t and prev.
            ab_t = self.alphas_cumprod[int(t_np)]
            ab_prev = self.alphas_cumprod[prev_np]
            alpha_sub = ab_t / ab_prev
            beta_sub = 1.0 - alpha_sub
            coef1 = beta_sub * torch.sqrt(ab_prev) / (1.0 - ab_t)
            coef2 = (1.0 - ab_prev) * torch.sqrt(alpha_sub) / (1.0 - ab_t)
            mean = coef1 * x0 + coef2 * xt
            var = beta_sub * (1.0 - ab_prev) / (1.0 - ab_t)
            noise = torch.randn(xt.shape, device=xt.device, dtype=xt.dtype, generator=generator)
            xt = mean + torch.sqrt(var) * noise
        return xt

    @torch.no_grad()
    def flow_sample(
        self,
        model: nn.Module,
        x_cond: torch.Tensor,
        a_src: torch.Tensor,
        a_tgt: torch.Tensor,
        steps: int = 50,
        cfg_scale: float = 1.0,
        sampler: str = "euler",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """JiT sampling: integrate dz/dt = v from t=0 (noise) to t=1 (data)."""
        model.eval()
        b = x_cond.shape[0]
        shape = (b, 1, x_cond.shape[-1])
        z = torch.randn(shape, device=x_cond.device, dtype=x_cond.dtype, generator=generator)
        z = z * self.noise_scale
        ts = np.linspace(0.0, 1.0, max(2, int(steps) + 1), dtype=np.float64)
        for i in range(len(ts) - 1):
            t0, t1 = float(ts[i]), float(ts[i + 1])
            t = torch.full((b,), t0, device=x_cond.device, dtype=x_cond.dtype)
            x_pred = self._model_x(
                model, z, t * self.time_scale, x_cond, a_src, a_tgt, cfg_scale
            )
            if self.clip_denoised is not None:
                c = float(self.clip_denoised)
                x_pred = x_pred.clamp(-c, c)
            v = (x_pred - z) / max(1.0 - t0, self.denom_clip)
            if sampler == "heun":
                z_pred = z + (t1 - t0) * v
                t_next = torch.full((b,), t1, device=x_cond.device, dtype=x_cond.dtype)
                x_pred2 = self._model_x(
                    model, z_pred, t_next * self.time_scale, x_cond, a_src, a_tgt, cfg_scale
                )
                if self.clip_denoised is not None:
                    x_pred2 = x_pred2.clamp(-float(self.clip_denoised), float(self.clip_denoised))
                v2 = (x_pred2 - z_pred) / max(1.0 - t1, self.denom_clip)
                z = z + (t1 - t0) * 0.5 * (v + v2)
            else:
                z = z + (t1 - t0) * v
        return z

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x_cond: torch.Tensor,
        a_src: torch.Tensor,
        a_tgt: torch.Tensor,
        steps: int = 50,
        cfg_scale: float = 1.0,
        sampler: str = "ddim",
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.formulation == "jit":
            if sampler not in ("euler", "heun"):
                raise ValueError("JiT sampling supports 'euler' or 'heun'")
            return self.flow_sample(
                model, x_cond, a_src, a_tgt, steps=steps, cfg_scale=cfg_scale,
                sampler=sampler, generator=generator,
            )
        if sampler == "ddpm":
            return self.ddpm_sample(
                model, x_cond, a_src, a_tgt, steps=steps, cfg_scale=cfg_scale, generator=generator
            )
        return self.ddim_sample(
            model, x_cond, a_src, a_tgt, steps=steps, cfg_scale=cfg_scale, eta=eta, generator=generator
        )
