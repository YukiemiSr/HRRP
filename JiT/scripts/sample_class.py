#!/usr/bin/env python3
"""Generate class-conditional HRRPs: (class label, target azimuth) -> HRRP.

No source HRRP is used.  Produces ``generated.npz`` (class/angle grid with real
 references) and ``sweep.npz`` (fixed class, target-angle sweep).

The class label is passed through the ``a_src`` slot of the shared generation
interface; ``model.cond_mode == 'class'`` interprets it as an integer label.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrrp_dit.data import (
    build_class_generation_set,
    build_class_sweep_set,
    load_split,
)
from hrrp_dit.generation import generate_profiles, load_generator
from hrrp_dit.utils import (
    ensure_dir,
    load_config,
    load_json,
    resolve_device,
    resolve_path,
    save_json,
    set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dit_buaa_class.yaml")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--sampler", default=None, choices=["ddim", "ddpm", "euler", "heun"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--ema", action="store_true")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config), args.overrides)
    ecfg = cfg.get("eval", {})
    seed = int(args.seed if args.seed is not None else ecfg.get("seed", 0))
    set_seed(seed)

    use_ema = args.ema or ((not args.no_ema) and bool(ecfg.get("use_ema", True)))
    if args.checkpoint:
        ckpt = resolve_path(args.checkpoint)
    else:
        run = resolve_path(cfg["train"].get("out_dir", "runs/dit_b_class"))
        ckpt = os.path.join(run, "best.pt" if use_ema else "last.pt")
    gen = load_generator(ckpt, device=resolve_device(args.device or "cuda:0"), use_ema=use_ema)
    if getattr(gen.model, "cond_mode", "") not in ("class", "class_cross"):
        raise SystemExit(
            f"checkpoint cond_mode={getattr(gen.model,'cond_mode',None)!r}, "
            "expected 'class' or 'class_cross'"
        )

    processed = resolve_path(cfg["data"]["processed_dir"])
    meta = load_json(os.path.join(processed, "meta.json"))
    class_names = list(meta["class_names"])
    split = load_split(processed, str(ecfg.get("split", "test")))
    out_dir = ensure_dir(resolve_path(ecfg.get("out_dir", "runs/dit_b_class/evaluation")))

    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    steps = int(args.steps if args.steps is not None else ecfg.get("steps", 50))
    cfg_scale = float(args.cfg_scale if args.cfg_scale is not None else ecfg.get("cfg_scale", 1.0))
    sampler = str(args.sampler or ecfg.get("sampler", "ddim"))
    batch_size = int(ecfg.get("batch_size", 128))

    print(f"[sample_class] checkpoint={ckpt} step={gen.step} cond_mode={getattr(gen.model, 'cond_mode', 'class')}")
    print(f"[sample_class] steps={steps} cfg_scale={cfg_scale} sampler={sampler}")

    data = build_class_generation_set(
        split,
        class_names,
        target_angles=ecfg.get("target_angles", [66.5, 69.0, 71.0, 79.0, 81.0, 84.0]),
        target_width=float(ecfg.get("target_width", 2.0)),
        n_per_angle=int(ecfg.get("n_per_angle", 200)),
        n_real=int(ecfg.get("n_real", 300)),
        classes=classes,
        seed=seed,
    )
    if len(data["y"]) == 0:
        raise RuntimeError("class generation set is empty")
    x_src = np.zeros((len(data["y"]), split.x.shape[1]), dtype=np.float32)
    a_src = data["y"].astype(np.float32)  # class labels in class mode
    x_gen = generate_profiles(
        gen, x_src, a_src, data["a_tgt"],
        batch_size=batch_size, steps=steps, cfg_scale=cfg_scale, sampler=sampler, seed=seed,
    )
    np.savez(
        os.path.join(out_dir, "generated.npz"),
        x_gen=x_gen,
        y=data["y"],
        a_tgt=data["a_tgt"],
        x_real=data["x_real"],
        real_y=data["real_y"],
        real_a_tgt=data["real_a_tgt"],
    )
    print(f"[sample_class] generated {len(x_gen)} profiles -> generated.npz")

    sweep_class = str(ecfg.get("sweep_class", "A321"))
    if sweep_class in class_names:
        sw = build_class_sweep_set(
            split,
            class_names,
            class_name=sweep_class,
            target_angles=ecfg.get("sweep_angles", [30, 40, 50, 60, 75, 90, 105, 120, 135, 150]),
            n_per_angle=int(ecfg.get("sweep_sources", 8)),
            seed=seed + 2,
        )
        x_src_s = np.zeros((len(sw["y"]), split.x.shape[1]), dtype=np.float32)
        a_src_s = sw["y"].astype(np.float32)
        x_gen_s = generate_profiles(
            gen, x_src_s, a_src_s, sw["a_tgt"],
            batch_size=batch_size, steps=steps, cfg_scale=cfg_scale, sampler=sampler, seed=seed + 2,
        )
        np.savez(
            os.path.join(out_dir, "sweep.npz"),
            x_gen=x_gen_s,
            y=sw["y"],
            a_tgt=sw["a_tgt"],
            class_name=np.array([sw["class_name"]]),
            sweep_angles=np.array(sw["sweep_angles"], np.float32),
        )
        print(f"[sample_class] generated {len(x_gen_s)} sweep profiles for {sweep_class}")

    save_json(
        os.path.join(out_dir, "generation_meta.json"),
        {
            "checkpoint": ckpt,
            "checkpoint_step": gen.step,
            "cond_mode": getattr(gen.model, "cond_mode", "class"),
            "steps": steps,
            "sampler": sampler,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "classes": classes or class_names,
            "target_angles": data["target_angles"],
            "counts": data["counts"],
            "signal_scale": gen.signal_scale,
        },
    )
    print(f"[sample_class] wrote {out_dir}")


if __name__ == "__main__":
    main()
