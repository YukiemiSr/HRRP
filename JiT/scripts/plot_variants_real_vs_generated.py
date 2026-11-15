#!/usr/bin/env python3
"""Compare Real HRRP and several v4 variants in separate panels.

Two modes:
  angles  : rows = Real + variants, columns = target azimuths, fixed class.
  classes : rows = Real + variants, columns = aircraft types, fixed angle.

Each panel is one individual HRRP (median L2 distance to the real class/angle
mean), so no real/generated overlay and no best/worst cherry-picking.

Examples:
  python3 scripts/plot_variants_real_vs_generated.py --mode angles \
      --class-name A321 --versions v4 v4b v4c v4e \
      --out runs/dit_b_class_stable/evaluation/variants_vs_real_angles.png

  python3 scripts/plot_variants_real_vs_generated.py --mode classes \
      --angle 79 --versions v4 v4b v4c v4e \
      --out runs/dit_b_class_stable/evaluation/variants_vs_real_classes.png
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
    "v4": "runs/dit_b_class_stable/evaluation/generated.npz",
    "v4b": "runs/dit_b_class_joint/evaluation/generated.npz",
    "v4c": "runs/dit_b_class_anglecross/evaluation/generated.npz",
    "v4e": "runs/dit_b_class_blockfilm/evaluation/generated.npz",
    "v5": "runs/dit_b_class_jit/evaluation/generated.npz",
    "v5cfg2": "runs/dit_b_class_jit/evaluation_cfg2/generated.npz",
    "v5p40": "runs/dit_b_class_jit_p40/evaluation/generated.npz",
    "v5p40cfg1": "runs/dit_b_class_jit_p40/evaluation_cfg1/generated.npz",
    "v5p40cfg2": "runs/dit_b_class_jit_p40/evaluation_cfg2/generated.npz",
}
SHORT = {
    "v4": "v4 (CFG4)",
    "v4b": "v4b (joint MLP)",
    "v4c": "v4c (angle cross)",
    "v4e": "v4e (block FiLM)",
    "v5": "v5 (CFG4)",
    "v5cfg2": "v5 (CFG2)",
    "v5p40": "v5-p40 (CFG4)",
    "v5p40cfg1": "v5-p40 (CFG1)",
    "v5p40cfg2": "v5-p40 (CFG2)",
}
COLORS = {
    "v4": "#D55E00",
    "v4b": "#0072B2",
    "v4c": "#009E73",
    "v4e": "#CC79A7",
    "v5": "#7B3294",
    "v5cfg2": "#1B7837",
    "v5p40": "#8C564B",
    "v5p40cfg1": "#2CA02C",
    "v5p40cfg2": "#17BECF",
}


def median_l2_sample(xs: np.ndarray, mean_profile: np.ndarray) -> np.ndarray:
    order = np.argsort(np.linalg.norm(xs - mean_profile, axis=1))
    return xs[int(order[len(order) // 2])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["angles", "classes"], required=True)
    ap.add_argument("--class-name", default="A321")
    ap.add_argument("--angles", type=float, nargs="+", default=[66.5, 69.0, 71.0, 79.0, 81.0, 84.0])
    ap.add_argument("--angle", type=float, default=79.0)
    ap.add_argument("--width", type=float, default=2.0)
    ap.add_argument("--versions", nargs="+", default=["v4", "v4b", "v4c", "v4e", "v5"],
                    choices=sorted(VERSION_RUNS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = load_json(resolve_path("data/buaa/meta.json"))
    names = list(meta["class_names"])
    test = load_split(resolve_path("data/buaa"), "test")
    zs = {v: np.load(resolve_path(VERSION_RUNS[v])) for v in args.versions}

    if args.mode == "angles":
        if args.class_name not in names:
            raise ValueError(f"unknown class {args.class_name}")
        cls = names.index(args.class_name)
        columns = [float(a) for a in args.angles]
        col_titles = [f"{a:g}°" for a in columns]
    else:
        # all classes, fixed target angle
        columns = list(range(len(names)))
        col_titles = names

    nrows = 1 + len(args.versions)
    ncols = len(columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.55 * nrows), squeeze=False)

    # Collect real and generated profiles first (per column), so each column
    # can share one y scale across Real and all variants.
    real_profiles, gen_profiles, used_titles = [], {v: [] for v in args.versions}, []
    for col in columns:
        if args.mode == "angles":
            cls = names.index(args.class_name)
            angle = float(col)
            title = f"{angle:g}°"
        else:
            cls = int(col)
            angle = float(args.angle)
            title = names[cls]

        rm = (test.y == cls) & (np.abs(test.angle - angle) <= args.width)
        if rm.sum() == 0:
            rm = (test.y == cls) & (np.abs(test.angle - angle) <= 2 * args.width)
        if rm.sum() == 0:
            print(f"skip {title}: no real profiles")
            continue
        R = test.x[rm]
        mu = R.mean(0)
        real_profiles.append(median_l2_sample(R, mu))
        ok = True
        for v in args.versions:
            gm = (zs[v]["y"] == cls) & (np.abs(zs[v]["a_tgt"] - angle) < 1e-4)
            if gm.sum() == 0:
                print(f"skip {title}: no {v} generated profiles")
                ok = False
                break
            gen_profiles[v].append(median_l2_sample(zs[v]["x_gen"][gm], mu))
        if ok:
            used_titles.append(title)

    ncols = len(used_titles)
    if ncols == 0:
        raise RuntimeError("no valid columns")
    # Rebuild the figure with the actual number of valid columns.
    plt.close(fig)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.55 * nrows), squeeze=False)

    for j in range(ncols):
        profiles = [real_profiles[j]] + [gen_profiles[v][j] for v in args.versions]
        ymax = max(float(p.max()) for p in profiles) * 1.12
        for row, prof in enumerate(profiles):
            color = "black" if row == 0 else COLORS[args.versions[row - 1]]
            ax = axes[row][j]
            ax.plot(prof, color=color, lw=1.0)
            ax.set_xlim(0, 799)
            ax.set_ylim(0, ymax)
            ax.set_xticks([0, 200, 400, 600, 800])
            ax.tick_params(labelbottom=(row == nrows - 1), labelleft=False, length=3)
            if row == nrows - 1:
                ax.set_xlabel("Range cell", fontsize=8)
            if row == 0:
                ax.set_title(used_titles[j], fontsize=10)

    axes[0][0].set_ylabel("Real HRRP", fontsize=9)
    for row, v in enumerate(args.versions, start=1):
        axes[row][0].set_ylabel(SHORT[v], fontsize=8)

    fig.subplots_adjust(left=0.065, right=0.995, top=0.93, bottom=0.12,
                        wspace=0.12, hspace=0.16)
    out = resolve_path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
