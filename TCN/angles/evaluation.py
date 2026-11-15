"""Independent validation/test on the checkpoint's registered angle split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..data import smoke_splits, split_hash
from ..utils import select_device
from .common import (
    AngleBatches,
    evaluate,
    load_checkpoint,
    restore_model,
    write_evaluation,
)
from .split import load_dataset, load_splits, read_setting


def run(args: argparse.Namespace, split: str) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("Only validation and test evaluation are supported.")
    cfg, path = read_setting(args.config)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    threads = cfg["runtime"]["threads"] if args.threads is None else args.threads
    batch = (
        cfg["training"]["batch_size"] if args.batch_size is None else args.batch_size
    )
    if min(threads, batch) < 1:
        raise ValueError("Threads and batch size must be positive.")
    torch.set_num_threads(threads)
    device = select_device(args.device or cfg["runtime"]["device"])
    dataset = load_dataset(cfg, path, args.data_dir)
    formal = load_splits(dataset, cfg, path)
    checkpoint = args.checkpoint.expanduser().resolve()
    saved = load_checkpoint(checkpoint)
    model = restore_model(saved, dataset, cfg, formal, device)
    p = saved["provenance"]
    if p["phase"] not in ("train", "smoke"):
        raise ValueError("Unknown checkpoint training phase.")
    training_splits = (
        smoke_splits(formal, dataset.targets) if p["phase"] == "smoke" else formal
    )
    if split_hash(training_splits) != p["actual_split_hash"]:
        raise ValueError("Checkpoint actual training split differs from setting.")
    is_smoke = args.smoke or p["phase"] == "smoke"
    selected = (
        smoke_splits(formal, dataset.targets)[split] if is_smoke else formal[split]
    )
    data = AngleBatches(dataset, {split: selected}, device, batch, p["conditioning"])
    metrics, truth, pred = evaluate(model, data, split, dataset.class_names)
    metadata = {
        "split": split,
        "phase": "smoke" if is_smoke else "full",
        "training_phase": p["phase"],
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": saved["epoch"],
        "conditioning": p["conditioning"],
        "bin_width_degrees": p["bin_width_degrees"],
        "split_hash": split_hash(formal),
        "actual_split_hash": p["actual_split_hash"],
        "test_used_for_selection": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    report = write_evaluation(
        output,
        split,
        dataset,
        selected,
        metrics,
        truth,
        pred,
        p["bin_width_degrees"],
        metadata,
    )
    print(
        f"{split}: phase={report['phase']} samples={len(selected)} Acc={metrics['accuracy']:.6f} output={output}",
        flush=True,
    )
    return report


def main(split: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args(), split)
