"""Matplotlib figures for generated HRRP views."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import FEATURE_LABELS, FEATURE_NAMES


def _mask_class_angle(y: np.ndarray, a_tgt: np.ndarray, c: int, angle: float) -> np.ndarray:
    return (y == int(c)) & (np.abs(a_tgt - float(angle)) < 1e-4)


def _panel(ax, x_real: np.ndarray, x_gen: np.ndarray, x_src: Optional[np.ndarray], title: str) -> None:
    mr = np.mean(x_real, axis=0)
    mg = np.mean(x_gen, axis=0)
    ax.plot(mr, color="#D55E00", lw=1.3, label="real mean")
    ax.plot(mg, color="#0072B2", lw=1.3, label="generated mean")
    if x_src is not None:
        ax.plot(x_src, color="0.55", lw=0.7, alpha=0.7, label="source")
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, len(mr) - 1)
    peak = max(mr.max(), mg.max(), 1e-6)
    ax.set_ylim(0.0, peak * 1.12)
    ax.tick_params(labelsize=6)


def plot_visual_grid(
    data: Dict[str, np.ndarray],
    class_names: Sequence[str],
    classes: Sequence[str],
    target_angles: Sequence[float],
    out_path: str,
) -> None:
    """One row per class, one column per target angle."""
    x_gen = data["x_gen"]
    x_real = data.get("x_real", np.zeros((0, x_gen.shape[1]), np.float32))
    y = data["y"]
    a_tgt = data["a_tgt"]
    real_y = data.get("real_y", np.zeros((0,), np.int64))
    real_a = data.get("real_a_tgt", np.zeros((0,), np.float32))
    x_src = data["x_src"] if "x_src" in data.files else None

    idx = [class_names.index(c) if isinstance(c, str) else int(c) for c in classes]
    nrow, ncol = len(idx), len(target_angles)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.65 * ncol, 1.9 * nrow), squeeze=False)
    for r, c in enumerate(idx):
        for k, ang in enumerate(target_angles):
            ax = axes[r][k]
            gm = _mask_class_angle(y, a_tgt, c, ang)
            rm = _mask_class_angle(real_y, real_a, c, ang)
            if gm.sum() == 0 or rm.sum() == 0:
                ax.axis("off")
                continue
            src = x_src[np.where(gm)[0][0]] if x_src is not None else None
            _panel(ax, x_real[rm], x_gen[gm], src, f"{class_names[c]} | {float(ang):g} deg")
            if r == 0 and k == 0:
                ax.legend(fontsize=5, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mean_profiles(
    data: Dict[str, np.ndarray],
    class_names: Sequence[str],
    target_angles: Sequence[float],
    out_path: str,
) -> None:
    """Rows = class, columns = target angle; real vs generated mean profiles."""
    x_gen = data["x_gen"]
    x_real = data["x_real"]
    y = data["y"]
    a_tgt = data["a_tgt"]
    real_y = data["real_y"]
    real_a = data["real_a_tgt"]
    nclass = len(class_names)
    ncol = len(target_angles)
    fig, axes = plt.subplots(nclass, ncol, figsize=(2.4 * ncol, 1.6 * nclass), squeeze=False)
    for c in range(nclass):
        for k, ang in enumerate(target_angles):
            ax = axes[c][k]
            gm = _mask_class_angle(y, a_tgt, c, ang)
            rm = _mask_class_angle(real_y, real_a, c, ang)
            if gm.sum() == 0 or rm.sum() == 0:
                ax.axis("off")
                continue
            mr = x_real[rm].mean(0)
            mg = x_gen[gm].mean(0)
            ax.plot(mr, color="#D55E00", lw=1.2)
            ax.plot(mg, color="#0072B2", lw=1.2)
            ax.set_title(f"{class_names[c]} {float(ang):g} deg", fontsize=6.5)
            ax.tick_params(labelsize=5)
            ax.set_xlim(0, len(mr) - 1)
            ax.set_ylim(0, max(mr.max(), mg.max(), 1e-6) * 1.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_sweep(sweep: Dict[str, np.ndarray], out_path: str) -> None:
    """Angle sweep for a fixed source: heatmap + selected profiles."""
    x_gen = sweep["x_gen"]
    a_tgt = sweep["a_tgt"]
    angles = np.unique(a_tgt)
    n_src = len(x_gen) // len(angles)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    # First source: angle x range heatmap.
    img = x_gen[: len(angles)]
    im = axes[0].imshow(
        img, aspect="auto", origin="lower", cmap="magma",
        extent=[0, img.shape[1], 0, len(angles)],
    )
    axes[0].set_yticks(np.arange(len(angles)) + 0.5)
    axes[0].set_yticklabels([f"{a:g}" for a in angles], fontsize=7)
    axes[0].set_xlabel("range cell")
    axes[0].set_ylabel("target azimuth (deg)")
    axes[0].set_title("generated HRRP of one source over azimuth")
    fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02)
    # Mean over sources for several angles.
    cmap = plt.get_cmap("viridis")
    for j, a in enumerate(angles):
        m = np.abs(a_tgt - a) < 1e-4
        axes[1].plot(x_gen[m].mean(0), color=cmap(j / max(len(angles) - 1, 1)), lw=1.0, label=f"{a:g}")
    axes[1].set_xlim(0, x_gen.shape[1] - 1)
    axes[1].set_xlabel("range cell")
    axes[1].set_title("mean generated profile vs target azimuth")
    axes[1].legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_kl(summary: Dict[str, Dict[str, float]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(FEATURE_NAMES))
    width = 0.38
    floor = [summary["real_vs_real"].get(n, np.nan) for n in FEATURE_NAMES]
    gen = [summary["generated"].get(n, np.nan) for n in FEATURE_NAMES]
    ax.bar(x - width / 2, floor, width, label="real vs real", color="#999999")
    ax.bar(x + width / 2, gen, width, label="generated vs real", color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_LABELS, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("histogram KL")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metrics_by_angle(rows: List[Dict[str, float]], out_path: str) -> None:
    """rows must contain 'target_angle', 'feature_kl', 'power_kl', 'mmd', 'mean_cosine'."""
    if not rows:
        return
    angles = sorted({float(r["target_angle"]) for r in rows})
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    specs = [
        ("feature_kl", "8-feature KL", "#0072B2"),
        ("power_kl", "scattering power KL", "#D55E00"),
        ("mmd", "profile MMD", "#009E73"),
        ("mean_cosine", "mean-profile cosine", "#CC79A7"),
    ]
    for ax, (key, title, color) in zip(axes, specs):
        vals = []
        for a in angles:
            v = [float(r[key]) for r in rows if abs(float(r["target_angle"]) - a) < 1e-6 and np.isfinite(r[key])]
            vals.append(float(np.mean(v)) if v else np.nan)
        ax.bar(np.arange(len(angles)), vals, color=color, alpha=0.85)
        ax.set_xticks(np.arange(len(angles)))
        ax.set_xticklabels([f"{a:g}" for a in angles], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_class_sweep(sweep, out_path: str) -> None:
    """Class-conditional sweep: mean generated profile at each target angle."""
    x_gen = sweep["x_gen"]
    a_tgt = sweep["a_tgt"]
    angles = sorted(np.unique(a_tgt).tolist())
    means = []
    for a in angles:
        mu = x_gen[np.abs(a_tgt - a) < 1e-4].mean(0)
        means.append(mu / (np.linalg.norm(mu) + 1e-12))
    means = np.stack(means, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    im = axes[0].imshow(
        means, aspect="auto", origin="lower", cmap="magma",
        extent=[0, means.shape[1], 0, len(angles)],
    )
    axes[0].set_yticks(np.arange(len(angles)) + 0.5)
    axes[0].set_yticklabels([f"{a:g}" for a in angles], fontsize=7)
    axes[0].set_xlabel("range cell")
    axes[0].set_ylabel("target azimuth (deg)")
    axes[0].set_title("mean generated HRRP vs target azimuth")
    fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02)
    cmap = plt.get_cmap("viridis")
    for j, (a, mu) in enumerate(zip(angles, means)):
        axes[1].plot(mu, color=cmap(j / max(len(angles) - 1, 1)), lw=1.0, label=f"{a:g}")
    axes[1].set_xlim(0, means.shape[1] - 1)
    axes[1].set_xlabel("range cell")
    axes[1].set_title("mean generated profile vs target azimuth")
    axes[1].legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
