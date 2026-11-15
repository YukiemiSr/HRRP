#!/usr/bin/env python3
"""Plot real vs generated HRRPs in separate rows (no overlay).

Layout follows the reference figure: columns are target azimuths, rows are
"Real HRRP" and one row per selected model.  Each panel is one individual
HRRP (not a mean), so fine peaks are visible.

Examples:
    # Real vs v4 (class-conditional, current task)
    python3 scripts/plot_real_vs_generated.py --versions v4 \
        --out runs/dit_b_class_stable/evaluation/real_vs_generated_v4.png

    # Real vs all models
    python3 scripts/plot_real_vs_generated.py \
        --versions v1 v2 v3b abs-v2 v4 \
        --out runs/real_vs_generated_all_versions.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrrp_dit.data import load_split
from hrrp_dit.utils import load_json, resolve_path

VERSION_RUNS = {
    "v1": "runs/dit_b_residual/evaluation/generated.npz",
    "v2": "runs/dit_b_residual_adaln/evaluation/generated.npz",
    "v3b": "runs/dit_b_residual_cross_v2/evaluation/generated.npz",
    "abs-v2": "runs/dit_b_absolute_v2/evaluation/generated.npz",
    "v4": "runs/dit_b_class_stable/evaluation/generated.npz",
    "v4b": "runs/dit_b_class_joint/evaluation/generated.npz",
    "v4c": "runs/dit_b_class_anglecross/evaluation/generated.npz",
    "v4e": "runs/dit_b_class_blockfilm/evaluation/generated.npz",
    "v5": "runs/dit_b_class_jit/evaluation/generated.npz",
}
VERSION_LABELS = {
    "v1": "DiT v1 (channel concat)",
    "v2": "DiT v2 (adaLN global)",
    "v3b": "DiT v3b (cross-attn)",
    "abs-v2": "DiT abs-v2 (absolute)",
    "v4": "DiT v4 (label + angle)",
    "v4b": "DiT v4b (joint class-angle)",
    "v4c": "DiT v4c (class adaLN + angle cross)",
    "v4e": "DiT v4e (block class-angle FiLM)",
    "v5": "DiT v5 (JiT x-pred)",
}
COLORS = {
    "v1": "#0072B2",
    "v2": "#009E73",
    "v3b": "#CC79A7",
    "abs-v2": "#E69F00",
    "v4": "#D55E00",
    "v4b": "#56B4E9",
    "v4c": "#009E73",
    "v4e": "#F0E442",
    "v5": "#7B3294",
}


def _median_by_cos(xs: np.ndarray, mean_profile: np.ndarray) -> np.ndarray:
    """Pick the sample with median L2 distance to the class/angle mean.

    L2 (rather than cosine) also accounts for amplitude, avoiding a
    low-amplitude outlier being selected as the "median" sample.
    """
    order = np.argsort(np.linalg.norm(xs - mean_profile, axis=1))
    return xs[int(order[len(order) // 2])]


def _real_pool(split, cls: int, angle: float, width: float):
    mask = (split.y == cls) & (np.abs(split.angle - float(angle)) <= float(width))
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"no real profiles for class {cls} near {angle} deg")
    return split.x[idx]


def _generated_pool(path: str, cls: int, angle: float) -> np.ndarray:
    z = np.load(path)
    mask = (z["y"] == cls) & (np.abs(z["a_tgt"] - float(angle)) < 1e-4)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"no generated profiles in {path} for class {cls} at {angle} deg")
    return z["x_gen"][idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-name", default="A321")
    ap.add_argument("--angles", type=float, nargs="+", default=[66.5, 69.0, 71.0, 79.0, 81.0, 84.0])
    ap.add_argument("--versions", nargs="+", default=["v4"], choices=sorted(VERSION_RUNS))
    ap.add_argument("--width", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = load_json(resolve_path("data/buaa/meta.json"))
    class_names = list(meta["class_names"])
    if args.class_name not in class_names:
        raise ValueError(f"unknown class {args.class_name}")
    cls = class_names.index(args.class_name)
    split = load_split(resolve_path("data/buaa"), "test")

    n_rows = 1 + len(args.versions)
    n_cols = len(args.angles)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.55 * n_cols, 1.55 * n_rows), squeeze=False
    )

    real_pools = [_real_pool(split, cls, a, args.width) for a in args.angles]
    real_means = [p.mean(0) for p in real_pools]
    real_profiles = [_median_by_cos(p, mu) for p, mu in zip(real_pools, real_means)]

    # Real row ---------------------------------------------------------
    for j, (a, x) in enumerate(zip(args.angles, real_profiles)):
        ax = axes[0][j]
        ax.plot(x, color="black", lw=1.1)
        ax.set_xlim(0, 799)
        ax.set_ylim(0, max(real_profiles[j].max(), 1e-6) * 1.12)
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.tick_params(labelbottom=False, labelleft=False, length=3)
        ax.set_title(f"{a:g}°", fontsize=10)

    # Generated rows ---------------------------------------------------
    for r, ver in enumerate(args.versions, start=1):
        path = resolve_path(VERSION_RUNS[ver])
        profiles = [
            _median_by_cos(_generated_pool(path, cls, a), mu)
            for a, mu in zip(args.angles, real_means)
        ]
        ymax = max(float(x.max()) for x in profiles)
        for j, (a, x) in enumerate(zip(args.angles, profiles)):
            ax = axes[r][j]
            ax.plot(x, color=COLORS[ver], lw=1.0)
            ax.set_xlim(0, 799)
            ax.set_ylim(0, ymax * 1.12)
            ax.set_xticks([0, 200, 400, 600, 800])
            ax.tick_params(labelbottom=(r == n_rows - 1), labelleft=False, length=3)
            if r == n_rows - 1:
                ax.set_xlabel("Range cell", fontsize=9)

    # Row labels on the first column (same style as the reference figure).
    short = {"v1": "v1 (concat)", "v2": "v2 (adaLN)", "v3b": "v3b (cross-attn)",
             "abs-v2": "abs-v2 (absolute)", "v4": "v4 (label + angle)",
             "v4b": "v4b (joint class-angle)",
             "v4c": "v4c (angle cross-attn)",
             "v4e": "v4e (block FiLM)",
             "v5": "v5 (JiT x-pred)"}
    row_labels = ["Real HRRP"] + [short[v] for v in args.versions]
    for r, label in enumerate(row_labels):
        axes[r][0].set_ylabel(label, fontsize=9, labelpad=6)

    fig.subplots_adjust(left=0.065, right=0.995, top=0.92, bottom=0.12,
                        wspace=0.18, hspace=0.22)
    out = resolve_path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
