"""From-scratch amplitude TCN, original 72/8/20 records, validation-only best selection."""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

if __package__:
    from .data import ResidentBatches, load_data, smoke_splits, split_hash
    from .model import TCNClassifier
    from .utils import (
        common_arguments,
        read_config,
        relative_path,
        seed_everything,
        select_device,
        write_json,
    )
    from .val import evaluate
else:
    from data import ResidentBatches, load_data, smoke_splits, split_hash
    from model import TCNClassifier
    from utils import (
        common_arguments,
        read_config,
        relative_path,
        seed_everything,
        select_device,
        write_json,
    )
    from val import evaluate


def run(args: argparse.Namespace) -> dict:
    cfg, path = read_config(args.config)
    training = cfg["training"]
    seed = training["seed"] if args.seed is None else args.seed
    epochs = (
        args.epochs
        if args.epochs is not None
        else 2 if args.smoke else training["epochs"]
    )
    batch = (
        args.batch_size
        if args.batch_size is not None
        else 8 if args.smoke else training["batch_size"]
    )
    if min(epochs, batch) < 1:
        raise ValueError("Epochs and batch size must be positive.")
    output = (
        args.output.resolve()
        if args.output
        else relative_path(path, cfg["runtime"]["output_root"])
        / f'{"smoke" if args.smoke else "train"}_seed_{seed}'
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    torch.set_num_threads(args.threads or cfg["runtime"]["threads"])
    seed_everything(seed)
    device = select_device(args.device or cfg["runtime"]["device"])
    dataset, formal = load_data(cfg, path, args.data_dir)
    splits = smoke_splits(formal, dataset.targets) if args.smoke else formal
    model = TCNClassifier(dataset.input_size, dataset.num_classes).to(device)
    # Preserve the original model initialization, shuffling and optimizer configuration.
    data = ResidentBatches(dataset, splits, device, batch)
    provenance = {
        "config": cfg,
        "phase": "smoke" if args.smoke else "train",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch,
        "device": str(device),
        "torch_version": torch.__version__,
        "model": model.description(),
        "source_signature": dataset.cache_metadata["source_signature"],
        "class_names": dataset.class_names,
        "split_hash": split_hash(formal),
        "actual_split_hash": split_hash(splits),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "formal_split_sizes": {k: len(v) for k, v in formal.items()},
        "from_scratch": True,
        "test_used_for_selection": False,
        "data_normalization": "per_record_l2_no_augmentation",
        "selection": "minimum_validation_loss",
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "split_indices.npz", **splits)
    write_json(output / "resolved_config.json", provenance)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed)
    history = []
    best_loss = float("inf")
    best_epoch = None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tic = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        count = 0
        for values, target in data.batches("train", generator):
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                training["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            loss_sum += loss.detach() * len(target)
            correct += (logits.detach().argmax(1) == target).sum()
            count += len(target)
        metrics, _, _ = evaluate(model, data, "validation", dataset.class_names)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "epoch": epoch,
            "streaming_train_loss": float(loss_sum) / count,
            "streaming_train_accuracy": float(correct) / count,
            "training_samples_seen": count,
            "validation": metrics,
            "epoch_seconds": time.perf_counter() - tic,
        }
        history.append(row)
        saved = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "provenance": provenance,
        }
        torch.save(saved, output / "last.pt")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            torch.save(saved, output / "best.pt")
        write_json(output / "history.json", history)
        print(
            f'epoch={epoch}/{epochs} train={row["streaming_train_accuracy"]:.4f} validation={metrics["accuracy"]:.4f} CE={metrics["loss"]:.5f} seconds={row["epoch_seconds"]:.2f}',
            flush=True,
        )
    result = {
        "model": "tcn",
        "phase": provenance["phase"],
        "seed": seed,
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "split_hash": split_hash(formal),
        "actual_split_hash": split_hash(splits),
        "test_used_for_selection": False,
    }
    # Test forward starts after all epochs and validation selection; no test-based checkpoint choice.
    for name in ("last", "best"):
        saved = torch.load(
            output / f"{name}.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(saved["model_state_dict"])
        result[f"{name}_test"], truth, pred = evaluate(
            model, data, "test", dataset.class_names
        )
        np.savez_compressed(
            output / f"{name}_test_predictions.npz",
            indices=splits["test"],
            labels=truth,
            predictions=pred,
        )
    result["total_seconds"] = time.perf_counter() - start
    write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 records/class/split and 2 epochs; workflow verification only.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
