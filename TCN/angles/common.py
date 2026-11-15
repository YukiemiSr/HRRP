"""Resident angle batches, portable checkpoint loading and audit metrics."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch

from ..data import ResidentBatches, split_hash
from ..utils import classification_metrics, write_json
from .backend import configure_backend
from .model import AngleTCN
from .split import write_csv, yaw_bins


def load_checkpoint(path: Path) -> dict:
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(path, **options)


def state_hash(model: torch.nn.Module, trunk_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if trunk_only and name.startswith("films."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AngleBatches(ResidentBatches):
    def __init__(
        self,
        dataset,
        splits: dict,
        device: torch.device,
        batch_size: int,
        conditioning: str,
    ) -> None:
        super().__init__(dataset, splits, device, batch_size)
        chosen = np.unique(np.concatenate(list(splits.values())))
        self.angles = None
        if conditioning != "none":
            columns = [
                (
                    dataset.pitch[chosen]
                    if conditioning == "pitch"
                    else dataset.yaw[chosen]
                )
            ]
            if conditioning == "yaw_pitch":
                columns.append(dataset.pitch[chosen])
            self.angles = torch.tensor(
                np.stack(columns, axis=1), dtype=torch.float32, device=device
            )

    def batches(self, split: str, generator: torch.Generator | None = None):
        ids = self.indices[split]
        if generator is not None:
            if split != "train":
                raise ValueError("Only train may be shuffled.")
            ids = ids[torch.randperm(len(ids), generator=generator).to(ids.device)]
        for part in ids.split(self.batch_size):
            yield self.signals[part], (
                self.angles[part] if self.angles is not None else None
            ), self.targets[part]


@torch.no_grad()
def evaluate(
    model: AngleTCN, data: AngleBatches, split: str, names: dict
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    labels, predictions = [], []
    total_loss = torch.zeros((), device=data.targets.device)
    for values, angles, target in data.batches(split):
        logits = model(values, angles)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits.")
        total_loss += torch.nn.functional.cross_entropy(logits, target, reduction="sum")
        labels.append(target)
        predictions.append(logits.argmax(1))
    truth = torch.cat(labels).cpu().numpy()
    pred = torch.cat(predictions).cpu().numpy()
    metrics = classification_metrics(truth, pred, names)
    metrics["loss"] = float(total_loss.item()) / len(truth)
    return metrics, truth, pred


def write_evaluation(
    output: Path,
    prefix: str,
    dataset,
    indices: np.ndarray,
    metrics: dict,
    labels: np.ndarray,
    pred: np.ndarray,
    width: int,
    metadata: dict,
    compact: bool = False,
    save_predictions: bool = True,
) -> dict:
    if len(indices) != len(labels) or not np.array_equal(
        dataset.targets[indices], labels
    ):
        raise ValueError("Prediction order/labels differ from the exact split indices.")
    names = [
        f"{prefix}.json",
        f"{prefix}_predictions.npz",
        f"{prefix}_per_class.csv",
        f"{prefix}_yaw_bins.csv",
        f"{prefix}_confusion.csv",
    ]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite evaluation artifacts.")
    bins = yaw_bins(dataset.yaw[indices], width)
    rows = []
    parity = 1 if metadata["split"] == "test" else 0
    for box in range(parity, 180 // width, 2):
        selected = bins == box
        count = int(selected.sum())
        correct = int((labels[selected] == pred[selected]).sum())
        rows.append(
            {
                "bin_index": box,
                "left_deg": box * width,
                "right_deg": (box + 1) * width,
                "samples": count,
                "correct": correct,
                "accuracy": correct / count if count else None,
                "classes_present": len(np.unique(labels[selected])),
            }
        )
    report = {
        **metadata,
        "metrics": metrics,
        "yaw_bins": rows,
        "prediction_storage": {
            "saved": save_predictions,
            "compact": compact,
            "angle_metadata_source_signature": dataset.cache_metadata[
                "source_signature"
            ],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        arrays = {"indices": indices, "labels": labels, "predictions": pred}
        if compact:
            for name, dtype in (
                ("indices", np.uint32),
                ("labels", np.uint8),
                ("predictions", np.uint8),
            ):
                narrowed = arrays[name].astype(dtype)
                if not np.array_equal(
                    narrowed.astype(arrays[name].dtype), arrays[name]
                ):
                    raise ValueError(
                        f"Compact dtype cannot losslessly represent {name}."
                    )
                arrays[name] = narrowed
        else:
            arrays.update(yaw=dataset.yaw[indices], pitch=dataset.pitch[indices])
        np.savez_compressed(output / names[1], **arrays)
    write_csv(
        output / names[2],
        [
            {"class_name": name, **values}
            for name, values in metrics["per_class"].items()
        ],
    )
    write_csv(output / names[3], rows)
    class_names = [dataset.class_names[i] for i in range(dataset.num_classes)]
    write_csv(
        output / names[4],
        [
            {
                "true_class": name,
                **{
                    f"pred_{label}": int(metrics["confusion_matrix"][i][j])
                    for j, label in enumerate(class_names)
                },
            }
            for i, name in enumerate(class_names)
        ],
    )
    write_json(output / names[0], report)
    return report


def restore_model(
    saved: dict, dataset, cfg: dict, formal: dict, device: torch.device
) -> AngleTCN:
    p = saved["provenance"]
    if (
        p["protocol"] != cfg["protocol"]
        or p["bin_width_degrees"] != cfg["split"]["bin_width_degrees"]
    ):
        raise ValueError("Checkpoint protocol/bin width differs from setting.")
    if p["source_signature"] != dataset.cache_metadata["source_signature"] or p[
        "split_hash"
    ] != split_hash(formal):
        raise ValueError("Checkpoint data source or frozen split differs.")
    if {int(k): v for k, v in p["class_names"].items()} != dataset.class_names:
        raise ValueError("Checkpoint class mapping differs.")
    if p["model"]["input_size"] != dataset.input_size:
        raise ValueError("Checkpoint input size differs.")
    configure_backend(p.get("cudnn_mode", "strict"))
    model = AngleTCN(dataset.input_size, dataset.num_classes, p["conditioning"])
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device)
