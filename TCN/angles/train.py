"""From-scratch TCN/FiLM training on a frozen alternating-yaw-bin split."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from ..data import smoke_splits, split_hash
from ..utils import relative_path, seed_everything, select_device, sha256, write_json
from .backend import MODES, configure_backend
from .common import (
    AngleBatches,
    evaluate,
    load_checkpoint,
    state_hash,
    write_evaluation,
)
from .model import AngleTCN, CONDITIONS
from .split import ROOT, load_dataset, load_splits, read_setting


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--conditioning", choices=CONDITIONS, default="yaw")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--cudnn-mode",
        choices=MODES,
        default="strict",
        help="fast avoids the legacy deterministic dilated-convolution fallback.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Save best weights only, small history and best-only compressed predictions.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two records/class/split, two epochs; workflow check only.",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    compact = args.compact
    cfg, path = read_setting(args.config)
    training = cfg["training"]
    if training["checkpoint_selection"] != "minimum_validation_loss":
        raise ValueError("Only validation CE may select the best checkpoint.")
    width = cfg["split"]["bin_width_degrees"]
    seed = training["seed"] if args.seed is None else args.seed
    epochs = (
        args.epochs
        if args.epochs is not None
        else (2 if args.smoke else training["epochs"])
    )
    batch = (
        args.batch_size
        if args.batch_size is not None
        else (8 if args.smoke else training["batch_size"])
    )
    threads = cfg["runtime"]["threads"] if args.threads is None else args.threads
    if min(epochs, batch, threads) < 1:
        raise ValueError("Epochs, batch size and thread count must be positive.")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else relative_path(path, cfg["runtime"]["output_root"])
        / ("smoke" if args.smoke else "train")
        / f"yaw{width}"
        / args.conditioning
        / f"seed_{seed}"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    torch.set_num_threads(threads)
    seed_everything(seed)
    # Seeding sets strict flags in the historical utilities, so apply policy AFTER it.
    backend = configure_backend(args.cudnn_mode)
    device = select_device(args.device or cfg["runtime"]["device"])
    dataset = load_dataset(cfg, path, args.data_dir)
    formal = load_splits(dataset, cfg, path)
    splits = smoke_splits(formal, dataset.targets) if args.smoke else formal
    model = AngleTCN(dataset.input_size, dataset.num_classes, args.conditioning).to(
        device
    )
    data = AngleBatches(dataset, splits, device, batch, args.conditioning)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed)
    provenance = {
        "protocol": cfg["protocol"],
        "bin_width_degrees": width,
        "conditioning": args.conditioning,
        "phase": "smoke" if args.smoke else "train",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch,
        "threads": threads,
        "cudnn_mode": args.cudnn_mode,
        "backend": backend,
        "device": str(device),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": cfg,
        "config_file": str(path),
        "source_signature": dataset.cache_metadata["source_signature"],
        "class_names": dataset.class_names,
        "split_hash": split_hash(formal),
        "actual_split_hash": split_hash(splits),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "formal_split_sizes": {k: len(v) for k, v in formal.items()},
        "model": model.description(),
        "initial_state_sha256": state_hash(model),
        "initial_trunk_sha256": state_hash(model, trunk_only=True),
        "from_scratch": True,
        "normalization": "per_record_l2",
        "sampling": "random_permutation_without_replacement",
        "loss": "classification_cross_entropy_only",
        "selection": "minimum_validation_loss",
        "test_used_for_selection": False,
        "storage_policy": "compact_best_weights_and_predictions" if compact else "full",
        "pitch_held_out": False,
        "updates_per_epoch": math.ceil(len(splits["train"]) / batch),
        "source_sha256": {
            str(p.relative_to(ROOT.parent)): sha256(p)
            for p in sorted(
                list(ROOT.glob("*.py"))
                + [
                    ROOT.parent / name
                    for name in (
                        "model.py",
                        "tcn.py",
                        "data.py",
                        "dataset.py",
                        "utils.py",
                    )
                ]
            )
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "split_indices.npz", **splits)
    write_json(output / "resolved_config.json", provenance)
    history = []
    best_loss, best_epoch = float("inf"), None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tic = time.perf_counter()
        model.train()
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        count, updates = 0, 0
        for values, angles, target in data.batches("train", generator):
            optimizer.zero_grad(set_to_none=True)
            logits = model(values, angles)
            loss = torch.nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            total_loss += loss.detach() * len(target)
            correct += (logits.detach().argmax(1) == target).sum()
            count += len(target)
            updates += 1
        validation, _, _ = evaluate(model, data, "validation", dataset.class_names)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history_validation = (
            {
                k: validation[k]
                for k in ("samples", "loss", "accuracy", "macro_accuracy", "macro_f1")
            }
            if compact
            else validation
        )
        row = {
            "epoch": epoch,
            "streaming_train_loss": float(total_loss.item()) / count,
            "streaming_train_accuracy": float(correct.item()) / count,
            "training_samples_seen": count,
            "updates": updates,
            "validation": history_validation,
            "epoch_seconds": time.perf_counter() - tic,
        }
        history.append(row)
        saved = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "provenance": provenance,
        }
        if not compact:
            saved["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(saved, output / "last.pt")
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save(saved, output / "best.pt.tmp")
            (output / "best.pt.tmp").replace(output / "best.pt")
        write_json(output / "history.json", history)
        print(
            f"yaw{width} {args.conditioning} epoch={epoch}/{epochs} train={row['streaming_train_accuracy']:.4f} val={validation['accuracy']:.4f} CE={validation['loss']:.5f}",
            flush=True,
        )
    result = {
        "protocol": cfg["protocol"],
        "bin_width_degrees": width,
        "conditioning": args.conditioning,
        "phase": provenance["phase"],
        "seed": seed,
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "split_hash": split_hash(formal),
        "actual_split_hash": split_hash(splits),
        "test_used_for_selection": False,
        "split_sizes": provenance["split_sizes"],
        "storage_policy": provenance["storage_policy"],
    }
    # First test forward happens only after ALL training epochs and validation selection.
    for name in ("last", "best"):
        checkpoint_epoch = epochs
        if name == "best" or not compact:
            saved = load_checkpoint(output / f"{name}.pt")
            model.load_state_dict(saved["model_state_dict"], strict=True)
            checkpoint_epoch = saved["epoch"]
        metrics, labels, pred = evaluate(model, data, "test", dataset.class_names)
        metadata = {
            "split": "test",
            "phase": "smoke" if args.smoke else "full",
            "checkpoint_epoch": checkpoint_epoch,
            "conditioning": args.conditioning,
            "bin_width_degrees": width,
            "split_hash": split_hash(formal),
            "actual_split_hash": split_hash(splits),
            "test_used_for_selection": False,
        }
        write_evaluation(
            output,
            f"{name}_test",
            dataset,
            splits["test"],
            metrics,
            labels,
            pred,
            width,
            metadata,
            compact=compact,
            save_predictions=not compact or name == "best",
        )
        result[f"{name}_test"] = metrics
    result["total_seconds"] = time.perf_counter() - start
    write_json(output / "result.json", result)
    write_json(
        output / "storage_summary.json",
        {
            "policy": provenance["storage_policy"],
            "saved_bytes_before_this_summary": sum(
                p.stat().st_size for p in output.iterdir() if p.is_file()
            ),
            "saved_files": {
                p.name: p.stat().st_size
                for p in sorted(output.iterdir())
                if p.is_file()
            },
        },
    )
    return result


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
