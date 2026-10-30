"""Local configuration, metrics and checkpoint utilities; no imports from the parent project."""

from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Mapping
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Mapping[int, str],
) -> Dict[str, object]:
    if len(labels) == 0:
        raise ValueError("Cannot evaluate an empty split.")
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    correct = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(
        correct, predicted, out=np.zeros_like(correct), where=predicted > 0
    )
    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(correct),
        where=(precision + recall) > 0,
    )
    per_class = {
        class_names[index]: {
            "class_index": index,
            "support": int(support[index]),
            "correct": int(correct[index]),
            "accuracy": float(recall[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index in range(num_classes)
    }
    return {
        "samples": int(len(labels)),
        "accuracy": float(np.mean(labels == predictions)),
        "macro_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: str | Path | None = None) -> tuple[dict, Path]:
    resolved = Path(path).resolve() if path else PACKAGE_ROOT / "config.json"
    return json.loads(resolved.read_text()), resolved


def relative_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (
        path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
    )


def select_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; use --device cpu or a CUDA-enabled environment."
        )
    return device


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.json")
    parser.add_argument(
        "--data-dir", type=Path, help="BUAA cache directory; overrides config."
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int)


def load_model(
    checkpoint: Path, dataset, formal_split_hash: str, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    if __package__:
        from .model import TCNClassifier
    else:
        from model import TCNClassifier
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    p = saved["provenance"]
    if p["source_signature"] != dataset.cache_metadata["source_signature"]:
        raise ValueError("Checkpoint and dataset source signatures differ.")
    if p["split_hash"] != formal_split_hash:
        raise ValueError("Checkpoint and frozen train/validation/test split differ.")
    if {int(k): v for k, v in p["class_names"].items()} != dataset.class_names:
        raise ValueError("Checkpoint class mapping differs from dataset.")
    model = TCNClassifier(dataset.input_size, dataset.num_classes)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device), saved
