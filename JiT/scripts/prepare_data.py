#!/usr/bin/env python3
"""Parse BUAA .dat recordings and cache file-level train/val/test arrays."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrrp_dit.data import prepare_dataset
from hrrp_dit.utils import load_config, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dit_buaa.yaml")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config), args.overrides)
    d = cfg["data"]
    summary = prepare_dataset(
        data_root=resolve_path(d["root"]),
        out_dir=resolve_path(d["processed_dir"]),
        subdir=d.get("subdir", "internal_data"),
        subdirs=d.get("subdirs"),
        label_json=resolve_path(d["label_json"]) if d.get("label_json") else None,
        class_order=d.get("class_order"),
        seed=int(cfg.get("seed", 42)),
        val_ratio=float(d.get("val_ratio", 0.1)),
        test_ratio=float(d.get("test_ratio", 0.1)),
        angle_field=str(d.get("angle_field", "yaw")),
        split_by=str(d.get("split_by", "profiles")),
        force=args.force,
    )
    print(json.dumps({
        "processed_dir": resolve_path(d["processed_dir"]),
        "signal_scale": summary["signal_scale"],
        "class_names": summary["class_names"],
        "split_records": summary["split_records"],
        "split_class_counts": summary["split_class_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
