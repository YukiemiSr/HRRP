"""Shared evaluation loop and independent validation-only entry point."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn

if __package__:
    from .data import ResidentBatches, load_data, smoke_splits, split_hash
    from .utils import (
        common_arguments,
        read_config,
        relative_path,
        select_device,
        classification_metrics,
        load_model,
        write_json,
        PACKAGE_ROOT,
    )
else:
    from data import ResidentBatches, load_data, smoke_splits, split_hash
    from utils import (
        common_arguments,
        read_config,
        relative_path,
        select_device,
        classification_metrics,
        load_model,
        write_json,
        PACKAGE_ROOT,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, data: ResidentBatches, split: str, class_names: dict
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    predictions = []
    loss_sum = torch.zeros((), device=data.targets.device)
    for values, target in data.batches(split):
        logits = model(values)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits.")
        loss_sum += nn.functional.cross_entropy(logits, target, reduction="sum")
        labels.append(target)
        predictions.append(logits.argmax(1))
    truth = torch.cat(labels).cpu().numpy()
    pred = torch.cat(predictions).cpu().numpy()
    metrics = classification_metrics(truth, pred, class_names)
    metrics["loss"] = float(loss_sum.item()) / len(truth)
    return metrics, truth, pred


def run_evaluation(args: argparse.Namespace, split: str) -> dict:
    cfg, path = read_config(args.config)
    threads = args.threads or cfg["runtime"]["threads"]
    torch.set_num_threads(threads)
    device = select_device(args.device or cfg["runtime"]["device"])
    batch = args.batch_size or cfg["training"]["batch_size"]
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint
        else relative_path(path, cfg["checkpoint"])
    )
    output = (
        args.output.resolve()
        if args.output
        else PACKAGE_ROOT
        / f'evaluations/{split}_{checkpoint.stem}{"_smoke" if args.smoke else ""}.json'
    )
    predictions = output.with_name(output.stem + "_predictions.npz")
    if output.exists() or predictions.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {predictions}")
    dataset, splits = load_data(cfg, path, args.data_dir)
    formal_hash = split_hash(splits)
    model, saved = load_model(checkpoint, dataset, formal_hash, device)
    selected = (
        smoke_splits(splits, dataset.targets)[split] if args.smoke else splits[split]
    )
    data = ResidentBatches(dataset, {split: selected}, device, batch)
    metrics, truth, pred = evaluate(model, data, split, dataset.class_names)
    result = {
        "split": split,
        "phase": "smoke" if args.smoke else "full",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": saved["epoch"],
        "training_phase": saved["provenance"].get("phase", "train"),
        "formal_split_hash": formal_hash,
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions, indices=selected, labels=truth, predictions=pred)
    write_json(output, result)
    print(
        f'{split}: epoch={saved["epoch"]}, records={metrics["samples"]}, Acc={metrics["accuracy"]:.6f}, Balanced Acc={metrics["macro_accuracy"]:.6f}, output={output}',
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only two records per class; not a formal score.",
    )
    run_evaluation(parser.parse_args(), "validation")


if __name__ == "__main__":
    main()
