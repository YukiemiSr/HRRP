"""Training loop for conditional HRRP DiT."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .data import (
    ClassBatcher,
    PairedBatcher,
    load_split,
    make_class_val_batch,
    make_val_pairs,
)
from .diffusion import GaussianDiffusion
from .identity import IdentityLoss
from .model import build_dit, count_parameters
from .utils import (
    AverageMeter,
    EMAModel,
    ensure_dir,
    get_logger,
    load_json,
    resolve_path,
    save_checkpoint,
    save_config,
    save_json,
)


def _autocast_dtype(name: str) -> Optional[torch.dtype]:
    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("off", "none", "fp32", "false", ""):
        return None
    raise ValueError(f"unknown amp mode {name!r}")


class Trainer:
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        resume: Optional[str] = None,
        reset_ema: bool = False,
    ):
        self.cfg = cfg
        self.device = device
        self.tcfg = cfg.get("train", {})
        self.out_dir = resolve_path(self.tcfg.get("out_dir", "runs/dit_buaa"))
        ensure_dir(self.out_dir)
        self.logger = get_logger(os.path.join(self.out_dir, "train.log"))

        processed = resolve_path(cfg["data"]["processed_dir"])
        self.meta = load_json(os.path.join(processed, "meta.json"))
        self.signal_scale = float(self.meta["signal_scale"])
        self.train_split = load_split(processed, "train")
        self.val_split = load_split(processed, "val")

        seed = int(cfg.get("seed", 42))
        dcfg = cfg.get("data", {})
        max_delta = dcfg.get("max_source_delta", 20.0)
        max_delta = None if max_delta is None else float(max_delta)
        self.target_mode = str(dcfg.get("target_mode", "absolute")).lower()
        if self.target_mode not in ("absolute", "residual"):
            raise ValueError(f"unknown data.target_mode {self.target_mode!r}")
        self.residual_scale = 1.0
        if self.target_mode == "residual":
            rscale = dcfg.get("residual_scale", None)
            if rscale is None:
                # Estimate the standard deviation of x(a_t)-x(a_o) on the same
                # pairing distribution used for training.
                tmp = PairedBatcher(
                    self.train_split.x,
                    self.train_split.y,
                    self.train_split.angle,
                    signal_scale=self.signal_scale,
                    batch_size=4096,
                    min_source_delta=float(dcfg.get("min_source_delta", 2.0)),
                    max_source_delta=max_delta,
                    p_self=0.0,
                    p_same_file=float(dcfg.get("p_same_file", 0.0)),
                    file_id=self.train_split.file_id,
                    target_mode="absolute",
                    seed=seed + 99,
                    device="cpu",
                )
                b = tmp.sample(4096)
                residual = (b["x"] - b["x_cond"]) * self.signal_scale
                rscale = float(residual.std().clamp_min(1e-6))
            self.residual_scale = float(rscale)
            self.logger.info("residual diffusion: scale %.6f", self.residual_scale)

        self.cond_mode = str(cfg.get("model", {}).get("cond_mode", "concat")).lower()
        val_n = int(self.tcfg.get("val_pairs", 1024))
        if self.cond_mode in ("class", "class_cross"):
            if self.target_mode != "absolute":
                raise ValueError("class-conditional mode requires data.target_mode='absolute'")
            self.batcher = ClassBatcher(
                self.train_split.x,
                self.train_split.y,
                self.train_split.angle,
                signal_scale=self.signal_scale,
                batch_size=int(self.tcfg.get("batch_size", 128)),
                seed=seed,
                device=device,
                class_balanced=bool(self.tcfg.get("class_balanced", False)),
            )
            self.val_batch = make_class_val_batch(
                self.val_split,
                signal_scale=self.signal_scale,
                n=val_n,
                seed=seed + 1,
                device=device,
            )
        else:
            self.batcher = PairedBatcher(
                self.train_split.x,
                self.train_split.y,
                self.train_split.angle,
                signal_scale=self.signal_scale,
                batch_size=int(self.tcfg.get("batch_size", 128)),
                min_source_delta=float(dcfg.get("min_source_delta", 2.0)),
                max_source_delta=max_delta,
                p_self=float(dcfg.get("p_self", 0.1)),
                p_same_file=float(dcfg.get("p_same_file", 0.0)),
                file_id=self.train_split.file_id,
                target_mode=self.target_mode,
                residual_scale=self.residual_scale,
                seed=seed,
                device=device,
            )
            self.val_batch = make_val_pairs(
                self.val_split,
                signal_scale=self.signal_scale,
                n=val_n,
                min_source_delta=float(dcfg.get("min_source_delta", 2.0)),
                max_source_delta=max_delta,
                p_self=0.0,
                p_same_file=0.5,
                target_mode=self.target_mode,
                residual_scale=self.residual_scale,
                seed=seed + 1,
                device=device,
            )

        self.model = build_dit(cfg).to(device)
        self.diffusion = GaussianDiffusion(cfg.get("diffusion", {})).to(device)
        self.ema = EMAModel(self.model, decay=float(self.tcfg.get("ema_decay", 0.9999)))

        self.identity_loss = None
        id_weight = float(self.tcfg.get("identity_loss_weight", 0.0))
        if id_weight > 0:
            id_ckpt = resolve_path(
                self.tcfg.get(
                    "identity_classifier",
                    cfg.get("classifier", {}).get("checkpoint", "checkpoints/classifier_buaa_full.pt"),
                )
            )
            self.identity_loss = IdentityLoss(
                id_ckpt,
                signal_scale=self.signal_scale,
                weight=id_weight,
                t_max=int(self.tcfg.get("identity_t_max", 400)),
                target_mode=self.target_mode,
                residual_scale=self.residual_scale,
                device=device,
            ).to(device)
            self.logger.info(
                "identity auxiliary loss: weight %.3f, t<=%d, classifier %s",
                id_weight, int(self.tcfg.get("identity_t_max", 400)), id_ckpt,
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.tcfg.get("lr", 1e-4)),
            betas=tuple(self.tcfg.get("betas", (0.9, 0.999))),
            weight_decay=float(self.tcfg.get("weight_decay", 0.0)),
        )
        self.amp_dtype = _autocast_dtype(str(self.tcfg.get("amp", "bf16")))
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.amp_dtype == torch.float16 and device.type == "cuda")
        )
        self.base_lr = float(self.tcfg.get("lr", 1e-4))
        self.warmup_steps = int(self.tcfg.get("warmup_steps", 1000))
        self.max_steps = int(self.tcfg.get("max_steps", 60000))
        self.grad_clip = float(self.tcfg.get("grad_clip", 1.0))
        self.cond_drop_prob = float(self.tcfg.get("cond_drop_prob", 0.1))
        self.step = 0
        self.best_val = float("inf")

        n_tokens = self.model.num_patches
        info = {
            "parameters": count_parameters(self.model),
            "tokens": n_tokens,
            "cond_mode": getattr(self.model, "cond_mode", "concat"),
            "patch_size": self.model.patch_size,
            "depth": len(self.model.blocks),
            "hidden_size": self.model.hidden_size,
            "signal_scale": self.signal_scale,
            "target_mode": self.target_mode,
            "residual_scale": self.residual_scale,
            "train_profiles": len(self.train_split),
            "val_profiles": len(self.val_split),
            "class_names": self.meta["class_names"],
        }
        save_json(os.path.join(self.out_dir, "model_info.json"), info)
        save_config(os.path.join(self.out_dir, "config.yaml"), cfg)
        self.logger.info(
            "model: %.1f M params | %d tokens | p=%d | depth=%d | d=%d",
            info["parameters"] / 1e6,
            n_tokens,
            self.model.patch_size,
            info["depth"],
            info["hidden_size"],
        )
        self.logger.info(
            "data: %d train / %d val profiles | signal_scale %.6f",
            len(self.train_split), len(self.val_split), self.signal_scale,
        )

        if resume:
            self._load(resume, reset_ema=reset_ema)

    # ------------------------------------------------------------------
    def _load(self, path: str, reset_ema: bool = False) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload.get("step", 0))
        if reset_ema or "ema" not in payload:
            # Restart the average from the loaded raw weights (useful when
            # resuming a short probe whose EMA is still near initialisation).
            self.ema = EMAModel(
                self.model, decay=float(self.tcfg.get("ema_decay", 0.9999))
            )
            self.best_val = float("inf")
            self.logger.info("EMA reset from loaded raw weights")
        else:
            self.ema.module.load_state_dict(payload["ema"])
            self.best_val = float(payload.get("extra", {}).get("best_val", self.best_val))
        self.logger.info("resumed from %s at step %d", path, self.step)

    def _lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)
        return self.base_lr

    def _p_self(self, step: int) -> float:
        """Curriculum for the fraction of self-reconstruction pairs.

        A high initial value forces the denoiser to actually encode the source
        HRRP (otherwise self-reconstruction is impossible); annealing to the
        final value then teaches azimuth translation instead of copying.
        """
        d = self.cfg.get("data", {})
        final = float(d.get("p_self", 0.1))
        initial = float(d.get("p_self_initial", final))
        hold = int(d.get("p_self_hold_steps", 0))
        anneal = int(d.get("p_self_anneal_steps", 0))
        if step < hold:
            return initial
        if anneal > 0 and step < hold + anneal:
            t = float(step - hold) / float(anneal)
            return initial + (final - initial) * t
        return final

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, use_ema: bool = True, repeats: int = 1) -> float:
        model = self.ema.module if use_ema else self.model
        was_training = model.training
        model.eval()
        losses = []
        for _ in range(max(1, int(repeats))):
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None and self.device.type == "cuda",
            ):
                loss, _ = self.diffusion.loss(model, self.val_batch, cond_drop_prob=0.0)
            losses.append(float(loss.detach()))
        model.train(was_training)
        return float(np.mean(losses))

    def _save(self, path: str, ema_only: bool = False) -> None:
        save_checkpoint(
            path,
            self.model,
            self.cfg,
            optimizer=None if ema_only else self.optimizer,
            ema=self.ema,
            step=self.step,
            extra={
                "signal_scale": self.signal_scale,
                "target_mode": self.target_mode,
                "residual_scale": self.residual_scale,
                "best_val": self.best_val,
                "ema_only": ema_only,
            },
        )

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, object]:
        tcfg = self.tcfg
        log_every = int(tcfg.get("log_every", 50))
        val_every = int(tcfg.get("val_every", 2000))
        save_every = int(tcfg.get("save_every", 5000))
        val_batches = int(tcfg.get("val_batches", 4))
        hist_path = os.path.join(self.out_dir, "history.jsonl")

        self.logger.info(
            "start training: step %d -> %d | batch %d | lr %.2e | amp %s | cond_drop %.2f",
            self.step, self.max_steps, self.batcher.batch_size, self.base_lr,
            self.amp_dtype, self.cond_drop_prob,
        )
        if self.step == 0:
            self._save(os.path.join(self.out_dir, "last.pt"))

        meter = AverageMeter()
        t0 = time.time()
        last_log = self.step
        while self.step < self.max_steps:
            lr = self._lr(self.step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            if hasattr(self.batcher, "p_self"):
                self.batcher.p_self = self._p_self(self.step)

            batch = self.batcher.sample()
            self.model.train()
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None and self.device.type == "cuda",
            ):
                loss, stats = self.diffusion.loss(
                    self.model,
                    batch,
                    cond_drop_prob=self.cond_drop_prob,
                    aux_loss=self.identity_loss,
                )
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            self.ema.update(self.model)
            self.step += 1
            meter.update(float(loss.detach()))

            if self.step % log_every == 0 or self.step == 1:
                dt = max(time.time() - t0, 1e-9)
                n = self.step - last_log
                id_ce = stats.get("identity_ce")
                id_txt = f"{float(id_ce):.3f}" if id_ce is not None else "-"
                self.logger.info(
                    "step %6d/%d | loss %.5f | id_ce %s | lr %.2e | p_self %.2f | %.3f s/step | %.0f prof/s | %.1f min",
                    self.step,
                    self.max_steps,
                    meter.avg,
                    id_txt,
                    lr,
                    getattr(self.batcher, "p_self", 0.0),
                    dt / max(n, 1),
                    n * self.batcher.batch_size / dt,
                    (time.time() - t0) / 60.0,
                )
                with open(hist_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "step": self.step,
                        "loss": meter.avg,
                        "lr": lr,
                        "sec_per_step": dt / max(n, 1),
                    }) + "\n")
                meter.reset()
                t0 = time.time()
                last_log = self.step

            if val_every > 0 and self.step % val_every == 0:
                val_loss = self.validate(use_ema=True, repeats=val_batches)
                self.logger.info("step %6d | val loss (EMA) %.5f | best %.5f", self.step, val_loss, self.best_val)
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save(os.path.join(self.out_dir, "best.pt"), ema_only=True)
                    self.logger.info("saved new best checkpoint (val %.5f)", val_loss)
                with open(hist_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": self.step, "val_loss_ema": val_loss}) + "\n")

            if save_every > 0 and self.step % save_every == 0:
                self._save(os.path.join(self.out_dir, "last.pt"))

        self._save(os.path.join(self.out_dir, "last.pt"))
        # Final validation / best selection.
        val_loss = self.validate(use_ema=True, repeats=val_batches)
        if val_loss < self.best_val:
            self.best_val = val_loss
            self._save(os.path.join(self.out_dir, "best.pt"), ema_only=True)
        self.logger.info("training finished at step %d | best val loss %.5f", self.step, self.best_val)
        return {"step": self.step, "best_val": self.best_val, "out_dir": self.out_dir}
