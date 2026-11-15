#!/usr/bin/env python3
"""Evaluate generated HRRP views: distribution metrics, identity, reconstruction."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrrp_dit.classifier import build_classifier
from hrrp_dit.data import load_split
from hrrp_dit.metrics import (
    FEATURE_NAMES,
    aggregate_metrics,
    feature_kl,
    feature_matrix,
    mean_profile_metrics,
    mmd_rbf,
    scattering_power_kl,
    select_cells_from_real,
)
from hrrp_dit.utils import (
    ensure_dir,
    load_config,
    load_json,
    resolve_device,
    resolve_path,
    save_json,
    set_seed,
)
from hrrp_dit.viz import (
    plot_feature_kl,
    plot_mean_profiles,
    plot_metrics_by_angle,
    plot_sweep,
    plot_visual_grid,
)


def _subsample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= 0 or len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=n, replace=False)]


def _metric_block(real: np.ndarray, fake: np.ndarray, max_n: int, seed: int) -> Dict[str, float]:
    real_m = _subsample(real, max_n, seed)
    fake_m = _subsample(fake, max_n, seed + 1)
    out: Dict[str, float] = {}
    fr = feature_matrix(real_m)
    fg = feature_matrix(fake_m)
    fkl = feature_kl(fr, fg)
    out.update({f"feat_{k}": v for k, v in fkl.items()})
    out["feature_kl"] = fkl["mean"]
    cells = select_cells_from_real(real)
    pkl, _ = scattering_power_kl(real, fake, cells)
    out["power_kl"] = pkl
    out.update(mean_profile_metrics(real, fake))
    out["mmd"] = mmd_rbf(real_m, fake_m, max_n=max_n, seed=seed)
    out["n_real"] = float(len(real))
    out["n_fake"] = float(len(fake))
    return out


def _baseline_block(real: np.ndarray, max_n: int, seed: int) -> Dict[str, float]:
    # Deterministic interleaved halves -> no RNG dependence.
    h1, h2 = real[0::2], real[1::2]
    if len(h1) < 2 or len(h2) < 2:
        return {"feature_kl": float("nan"), "power_kl": float("nan"), "mmd": float("nan")}
    return _metric_block(h1, h2, max_n=max_n, seed=seed + 2)


@torch.no_grad()
def _predict(model, x: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).unsqueeze(1).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dit_buaa.yaml")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--generated-dir", default=None)
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config), args.overrides)
    ecfg = cfg.get("eval", {})
    set_seed(args.seed)
    processed = resolve_path(cfg["data"]["processed_dir"])
    meta = load_json(os.path.join(processed, "meta.json"))
    class_names = list(meta["class_names"])
    gen_dir = resolve_path(args.generated_dir or ecfg.get("out_dir", "runs/dit_b_residual/evaluation"))
    out_dir = ensure_dir(gen_dir)
    max_n = int(args.max_samples if args.max_samples is not None else ecfg.get("max_metrics_samples", 256))
    device = resolve_device(str(args.device or cfg["train"].get("device", "cuda:0")))

    z = np.load(os.path.join(gen_dir, "generated.npz"))
    has_source = "x_src" in z.files
    x_gen, x_real = z["x_gen"], z["x_real"]
    y, a_tgt = z["y"], z["a_tgt"]
    real_y, real_a = z["real_y"], z["real_a_tgt"]
    print(
        f"[eval] mode={'source' if has_source else 'class'} | "
        f"generated {len(x_gen)} profiles | {len(x_real)} real references"
    )

    rows: List[Dict[str, object]] = []
    baseline_rows: List[Dict[str, object]] = []
    copy_rows: List[Dict[str, object]] = []
    x_src_all = z["x_src"] if has_source else None
    for c in sorted(np.unique(y).tolist()):
        cname = class_names[int(c)]
        angles = sorted(np.unique(a_tgt[y == c]).tolist())
        for a in angles:
            gm = (y == c) & (np.abs(a_tgt - a) < 1e-4)
            rm = (real_y == c) & (np.abs(real_a - a) < 1e-4)
            if gm.sum() < 2 or rm.sum() < 4:
                continue
            m = _metric_block(x_real[rm], x_gen[gm], max_n=max_n, seed=args.seed + int(c))
            b = _baseline_block(x_real[rm], max_n=max_n, seed=args.seed + int(c))
            rows.append({"class": cname, "class_idx": int(c), "target_angle": float(a), "metrics": m})
            baseline_rows.append({"class": cname, "class_idx": int(c), "target_angle": float(a), "metrics": b})
            if has_source:
                # "nearest acquired view" baseline: copy the source HRRP.
                mc = _metric_block(x_real[rm], x_src_all[gm], max_n=max_n, seed=args.seed + int(c))
                copy_rows.append({"class": cname, "class_idx": int(c), "target_angle": float(a), "metrics": mc})

    if not rows:
        raise RuntimeError("no valid class/angle groups to evaluate")

    # Aggregates ---------------------------------------------------------
    overall_gen = aggregate_metrics([r["metrics"] for r in rows])
    overall_base = aggregate_metrics([r["metrics"] for r in baseline_rows])
    overall_copy = aggregate_metrics([r["metrics"] for r in copy_rows]) if copy_rows else {}
    per_class: Dict[str, Dict[str, object]] = {}
    for cname in class_names:
        gr = [r["metrics"] for r in rows if r["class"] == cname]
        br = [r["metrics"] for r in baseline_rows if r["class"] == cname]
        cr = [r["metrics"] for r in copy_rows if r["class"] == cname]
        if gr:
            entry = {
                "generated": aggregate_metrics(gr),
                "real_vs_real": aggregate_metrics(br),
                "n_groups": len(gr),
            }
            if has_source:
                entry["copy_source"] = aggregate_metrics(cr)
            per_class[cname] = entry
    per_angle: Dict[str, Dict[str, object]] = {}
    for a in sorted({float(r["target_angle"]) for r in rows}):
        gr = [r["metrics"] for r in rows if abs(float(r["target_angle"]) - a) < 1e-6]
        br = [r["metrics"] for r in baseline_rows if abs(float(r["target_angle"]) - a) < 1e-6]
        cr = [r["metrics"] for r in copy_rows if abs(float(r["target_angle"]) - a) < 1e-6]
        entry = {
            "generated": aggregate_metrics(gr),
            "real_vs_real": aggregate_metrics(br),
        }
        if has_source:
            entry["copy_source"] = aggregate_metrics(cr)
        per_angle[f"{a:g}"] = entry

    # Identity preservation ---------------------------------------------
    identity: Dict[str, object] = {}
    ckpt_path = resolve_path(args.classifier or cfg["classifier"].get("checkpoint", "checkpoints/classifier_buaa_full.pt"))
    if os.path.isfile(ckpt_path):
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        ck_cfg = payload.get("config", cfg)
        ck_cfg.setdefault("classifier", {})
        ck_cfg["classifier"]["num_classes"] = len(class_names)
        clf = build_classifier(ck_cfg).to(device)
        clf.load_state_dict(payload["model"])
        pred = _predict(clf, x_gen, device)
        identity["generated_accuracy"] = float((pred == y).mean())
        identity["classifier_val_acc"] = float(payload.get("val_acc", float("nan")))
        identity["per_class"] = {
            class_names[c]: float((pred[y == c] == c).mean())
            for c in range(len(class_names)) if (y == c).any()
        }
        test = load_split(processed, "test")
        pred_real = _predict(clf, test.x, device)
        identity["real_test_accuracy"] = float((pred_real == test.y).mean())

        if has_source:
            # Identity metrics that do not depend on the classifier being
            # accurate on the generated distribution:
            #  * prediction consistency: generated and source get the same label;
            #  * cosine to the own source vs. a random source of another class.
            pred_src = _predict(clf, x_src_all, device)
            identity["source_accuracy"] = float((pred_src == y).mean())
            identity["prediction_consistency"] = float((pred == pred_src).mean())
            own_cos = np.sum(x_gen * x_src_all, axis=1)
            rng = np.random.default_rng(args.seed + 17)
            other_cos = np.empty_like(own_cos)
            for i in range(len(x_gen)):
                cand = np.where(y != y[i])[0]
                j = int(rng.choice(cand))
                other_cos[i] = float(np.sum(x_gen[i] * x_src_all[j]))
            identity["cos_own_source"] = float(own_cos.mean())
            identity["cos_other_class_source"] = float(other_cos.mean())
            identity["frac_own_source_closer"] = float((own_cos > other_cos).mean())
            print(
                f"[eval] identity: generated {identity['generated_accuracy']:.4f} | "
                f"real {identity['real_test_accuracy']:.4f} | consistency "
                f"{identity['prediction_consistency']:.4f} | cos(own src) "
                f"{identity['cos_own_source']:.4f} vs other {identity['cos_other_class_source']:.4f}"
            )
        else:
            identity["mode"] = "class_conditional"
            print(
                f"[eval] class identity: generated top-1 "
                f"{identity['generated_accuracy']:.4f} | real test "
                f"{identity['real_test_accuracy']:.4f}"
            )
    else:
        identity["error"] = f"classifier checkpoint not found: {ckpt_path}"
        print(f"[eval] warning: {identity['error']}")

    # Self-reconstruction ------------------------------------------------
    self_recon: Dict[str, float] = {}
    sr_path = os.path.join(gen_dir, "self_recon.npz")
    if os.path.isfile(sr_path):
        s = np.load(sr_path)
        xg, xs = s["x_gen"], s["x_src"]
        cos = np.sum(xg * xs, axis=1)
        l1 = np.abs(xg - xs).sum(axis=1)
        rel = np.linalg.norm(xg - xs, axis=1) / (np.linalg.norm(xs, axis=1) + 1e-12)
        self_recon = {
            "cosine_mean": float(cos.mean()),
            "cosine_median": float(np.median(cos)),
            "l1_mean": float(l1.mean()),
            "rel_l2_mean": float(rel.mean()),
            "n": int(len(cos)),
        }
        print(f"[eval] self-reconstruction cosine {self_recon['cosine_mean']:.4f}")

    # Azimuth sweep ------------------------------------------------------
    sweep_metrics: Dict[str, object] = {}
    sw_path = os.path.join(gen_dir, "sweep.npz")
    if os.path.isfile(sw_path):
        s = np.load(sw_path, allow_pickle=True)
        at = s["a_tgt"]
        if has_source:
            xs = s["x_src"]
            # Group equal source rows (the sweep repeats each source over angles).
            _, inv = np.unique(xs, axis=0, return_inverse=True)
            adjacent, endpoints, same_angle = [], [], []
            for k in np.unique(inv):
                m = inv == k
                order = np.argsort(at[m])
                g = s["x_gen"][m][order]
                aa = at[m][order]
                if len(g) >= 2:
                    adjacent.extend(np.sum(g[1:] * g[:-1], axis=1).tolist())
                    endpoints.append(float(np.sum(g[0] * g[-1])))
                # Reconstruction when the requested angle equals the source angle.
                for i, a in enumerate(aa):
                    if abs(a - float(s["source_angle"][0])) < 1e-4:
                        same_angle.append(float(np.sum(g[i] * xs[np.where(inv == k)[0][0]])))
            sweep_metrics = {
                "adjacent_cosine": float(np.mean(adjacent)) if adjacent else float("nan"),
                "endpoint_cosine": float(np.mean(endpoints)) if endpoints else float("nan"),
                "same_angle_cosine": float(np.mean(same_angle)) if same_angle else float("nan"),
                "n_sources": int(len(np.unique(inv))),
                "angles": sorted(np.unique(at).tolist()),
            }
        else:
            # Class-conditional sweep: compare mean generated profiles at
            # adjacent target angles (sample noise averages out).
            angles = sorted(np.unique(at).tolist())
            means = []
            for a in angles:
                m = np.abs(at - a) < 1e-4
                mu = s["x_gen"][m].mean(0)
                means.append(mu / (np.linalg.norm(mu) + 1e-12))
            if len(means) >= 2:
                adjacent = [float(np.dot(means[i], means[i + 1])) for i in range(len(means) - 1)]
                endpoint = float(np.dot(means[0], means[-1]))
            else:
                adjacent, endpoint = [], float("nan")
            sweep_metrics = {
                "adjacent_cosine": float(np.mean(adjacent)) if adjacent else float("nan"),
                "endpoint_cosine": endpoint,
                "same_angle_cosine": float("nan"),
                "n_samples_per_angle": int(len(s["x_gen"]) // max(len(angles), 1)),
                "class_name": str(s["class_name"][0]) if "class_name" in s.files else None,
                "angles": angles,
            }
        print(f"[eval] sweep adjacent cosine {sweep_metrics['adjacent_cosine']:.4f}")

    # Save JSON ----------------------------------------------------------
    overall_entry = {"generated": overall_gen, "real_vs_real": overall_base}
    if has_source:
        overall_entry["copy_source"] = overall_copy
    results = {
        "protocol": {
            "generated_dir": gen_dir,
            "mode": "source" if has_source else "class_conditional",
            "checkpoint_step": load_json(os.path.join(gen_dir, "generation_meta.json"))["checkpoint_step"]
            if os.path.isfile(os.path.join(gen_dir, "generation_meta.json")) else None,
            "max_metrics_samples": max_n,
            "class_names": class_names,
            "n_class_angle_groups": len(rows),
        },
        "overall": overall_entry,
        "per_class": per_class,
        "per_angle": per_angle,
        "per_class_angle": rows,
        "identity": identity,
        "self_recon": self_recon,
        "sweep": sweep_metrics,
    }
    save_json(os.path.join(out_dir, "metrics.json"), results)

    # Figures ------------------------------------------------------------
    viz_classes = ecfg.get("viz_classes", class_names[:3])
    target_angles = ecfg.get("target_angles", sorted({float(r["target_angle"]) for r in rows}))
    plot_visual_grid(z, class_names, viz_classes, target_angles, os.path.join(out_dir, "visual_grid.png"))
    plot_mean_profiles(z, class_names, target_angles, os.path.join(out_dir, "mean_profiles.png"))
    if sweep_metrics:
        if has_source:
            plot_sweep(np.load(sw_path), os.path.join(out_dir, "angle_sweep.png"))
        else:
            from hrrp_dit.viz import plot_class_sweep

            plot_class_sweep(np.load(sw_path, allow_pickle=True), os.path.join(out_dir, "angle_sweep.png"))
    feat_summary = {
        "real_vs_real": {n: overall_base.get(f"feat_{n}", float("nan")) for n in FEATURE_NAMES},
        "generated": {n: overall_gen.get(f"feat_{n}", float("nan")) for n in FEATURE_NAMES},
    }
    plot_feature_kl(feat_summary, os.path.join(out_dir, "feature_kl.png"))
    viz_rows = [{"target_angle": r["target_angle"], **r["metrics"]} for r in rows]
    plot_metrics_by_angle(viz_rows, os.path.join(out_dir, "metrics_by_angle.png"))

    # Markdown summary ---------------------------------------------------
    def _fmt(d: Dict[str, float]) -> str:
        return (
            f"{d.get('feature_kl', float('nan')):.3f} | "
            f"{d.get('power_kl', float('nan')):.3f} | "
            f"{d.get('mmd', float('nan')):.4f} | "
            f"{d.get('mean_cosine', float('nan')):.4f} | "
            f"{d.get('mean_rel_l2', float('nan')):.3f}"
        )

    lines = [
        "# DiT HRRP view synthesis — evaluation",
        "",
        f"Generated directory: `{gen_dir}`",
        f"Groups: {len(rows)} class/angle cells; metrics subsampled to at most {max_n} profiles.",
        "",
        "## Overall",
        "",
        "| Data | 8-feature KL ↓ | Scattering power KL ↓ | MMD ↓ | Mean-profile cosine ↑ | Mean-profile rel-L2 ↓ |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Real vs real (noise floor) | {_fmt(overall_base)} |",
    ]
    if has_source:
        lines.append(f"| Copy source (nearest acquired view) | {_fmt(overall_copy)} |")
    lines += [
        f"| **Generated vs real** | {_fmt(overall_gen)} |",
        "",
        "## Per class (generated vs real)",
        "",
        "| Class | 8-feature KL | Power KL | MMD | Mean cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for cname in class_names:
        if cname not in per_class:
            continue
        g = per_class[cname]["generated"]
        lines.append(
            f"| {cname} | {g.get('feature_kl', float('nan')):.3f} | {g.get('power_kl', float('nan')):.3f} "
            f"| {g.get('mmd', float('nan')):.4f} | {g.get('mean_cosine', float('nan')):.4f} |"
        )
    lines += [
        "",
        "## Per target azimuth (generated vs real)",
        "",
        "| Azimuth | 8-feature KL | Power KL | MMD | Mean cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for a in sorted(per_angle, key=float):
        g = per_angle[a]["generated"]
        lines.append(
            f"| {a}° | {g.get('feature_kl', float('nan')):.3f} | {g.get('power_kl', float('nan')):.3f} "
            f"| {g.get('mmd', float('nan')):.4f} | {g.get('mean_cosine', float('nan')):.4f} |"
        )
    if identity and "generated_accuracy" in identity:
        lines += [
            "",
            "## Identity preservation",
            "",
            f"* Generated top-1 accuracy: **{identity['generated_accuracy']:.4f}**",
            f"* Real test top-1 accuracy: **{identity['real_test_accuracy']:.4f}**",
        ]
        if has_source:
            lines += [
                f"* Source top-1 accuracy: {identity.get('source_accuracy', float('nan')):.4f}",
                f"* Prediction consistency (generated == source): "
                f"**{identity.get('prediction_consistency', float('nan')):.4f}**",
                f"* Cosine to own source: {identity.get('cos_own_source', float('nan')):.4f} "
                f"(random other-class source: {identity.get('cos_other_class_source', float('nan')):.4f})",
                f"* Fraction where own source is closer: "
                f"{identity.get('frac_own_source_closer', float('nan')):.4f}",
            ]
        else:
            lines.append("* 类别条件模式：top-1 是“生成样本被分类为所请求类别”的准确率。")
    if self_recon:
        lines += [
            "",
            "## Self-reconstruction (a_tgt = a_src)",
            "",
            f"* Cosine: {self_recon['cosine_mean']:.4f} (median {self_recon['cosine_median']:.4f})",
            f"* L1: {self_recon['l1_mean']:.4f} | relative L2: {self_recon['rel_l2_mean']:.4f}",
        ]
    if sweep_metrics:
        lines += [
            "",
            "## Azimuth sweep",
            "",
            f"* Adjacent-angle cosine: {sweep_metrics['adjacent_cosine']:.4f} "
            "(1.0 would mean the target azimuth is ignored)",
            f"* Endpoint cosine: {sweep_metrics['endpoint_cosine']:.4f}",
        ]
        if np.isfinite(sweep_metrics.get("same_angle_cosine", float("nan"))):
            lines.append(
                f"* Same-angle reconstruction cosine: {sweep_metrics['same_angle_cosine']:.4f}"
            )
    lines += [
        "",
        "## Figures",
        "",
        "![visual grid](visual_grid.png)",
        "",
        "![mean profiles](mean_profiles.png)",
        "",
        "![angle sweep](angle_sweep.png)",
        "",
        "![feature KL](feature_kl.png)",
        "",
        "![metrics by angle](metrics_by_angle.png)",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(json.dumps({
        "overall_generated": overall_gen,
        "overall_real_vs_real": overall_base,
        "overall_copy_source": overall_copy,
        "identity": identity,
        "self_recon": self_recon,
        "sweep": sweep_metrics,
    }, indent=2, ensure_ascii=False))
    print(f"[eval] wrote {out_dir}")


if __name__ == "__main__":
    main()
