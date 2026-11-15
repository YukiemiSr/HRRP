"""Configuration, logging, checkpoint, EMA and HRRP post-processing helpers."""
from __future__ import annotations

import copy
import json
import logging
import os
import random
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML config and apply ``key.subkey=value`` overrides."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for item in overrides or []:
        _apply_override(cfg, item)
    return cfg


def _parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _apply_override(cfg: Dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override must look like key.subkey=value, got {item!r}")
    key, value = item.split("=", 1)
    node: Any = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = _parse_scalar(value)


def save_config(path: str, cfg: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Logging / IO
# ---------------------------------------------------------------------------
def get_logger(log_file: Optional[str] = None, name: str = "hrrp_dit") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        ensure_dir(os.path.dirname(os.path.abspath(log_file)))
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path: str) -> str:
    """Resolve a config path relative to the project root."""
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    model: nn.Module,
    cfg: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional["EMAModel"] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "config": cfg,
        "step": int(step),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if ema is not None:
        payload["ema"] = ema.module.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str, map_location: str | torch.device = "cpu"
) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# EMA (DiT paper: decay 0.9999, all reported samples use the EMA model)
# ---------------------------------------------------------------------------
class EMAModel:
    """Exponential moving average of a model's parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for e, m in zip(self.module.parameters(), model.parameters()):
            e.mul_(d).add_(m.detach().to(e.dtype), alpha=1.0 - d)
        for e, m in zip(self.module.buffers(), model.buffers()):
            e.copy_(m)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.module.state_dict()


# ---------------------------------------------------------------------------
# HRRP post-processing
# ---------------------------------------------------------------------------
def scaled_to_hrrp(x_scaled: torch.Tensor, scale: float) -> torch.Tensor:
    """Map network output in scaled units back to a unit-L2 HRRP.

    HRRP amplitudes are non-negative; predictions are clipped, rescaled and
    L2-normalised exactly like the measured profiles used for training.
    """
    x = x_scaled * float(scale)
    x = x.clamp_min(0.0)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def hrrp_to_scaled(x: np.ndarray | torch.Tensor, scale: float) -> torch.Tensor:
    """Unit-L2 HRRP -> network units (global std ~ 1)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.float() / float(scale)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
