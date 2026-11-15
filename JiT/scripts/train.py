#!/usr/bin/env python3
"""Train a conditional HRRP DiT."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrrp_dit.data import prepare_dataset
from hrrp_dit.trainer import Trainer
from hrrp_dit.utils import load_config, resolve_device, resolve_path, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dit_buaa.yaml")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--reset-ema", action="store_true", help="re-initialise EMA from raw weights")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config), args.overrides)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
    if args.device is not None:
        cfg["train"]["device"] = args.device
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg["train"].get("device", "cuda:0")))

    d = cfg["data"]
    out_dir = resolve_path(d["processed_dir"])
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.isfile(meta_path):
        print("[train] processed data missing; preparing it first")
        prepare_dataset(
            data_root=resolve_path(d["root"]),
            out_dir=out_dir,
            subdir=d.get("subdir", "internal_data"),
            seed=int(cfg.get("seed", 42)),
            val_ratio=float(d.get("val_ratio", 0.1)),
            test_ratio=float(d.get("test_ratio", 0.1)),
            angle_field=str(d.get("angle_field", "yaw")),
            split_by=str(d.get("split_by", "profiles")),
        )

    trainer = Trainer(cfg, device=device, resume=args.resume, reset_ema=args.reset_ema)
    result = trainer.train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
