"""Registered 1:1 yaw bins; one immutable, shared split per bin width."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from ..data import split_hash, validate_splits
from ..dataset import BUAAInternalDataset, stratified_train_test_split
from ..utils import relative_path, sha256, write_json

ROOT = Path(__file__).resolve().parent
WIDTHS = (5, 10, 15)


def read_setting(path: Path) -> tuple[dict, Path]:
    path = path.expanduser().resolve()
    cfg = json.loads(path.read_text())
    s = cfg["split"]
    if (
        cfg["protocol"] != "alternating_yaw_bins_1to1_v1"
        or s["bin_width_degrees"] not in WIDTHS
        or s["origin_degrees"] != 0
        or s["period_bins"] != 2
        or s["test_bin_offset"] != 1
        or s["validation_fraction_of_train_pool"] != 0.1
        or s["seed"] != 56
    ):
        raise ValueError("Setting must use registered 5/10/15-degree alternating bins.")
    return cfg, path


def yaw_bins(yaw: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(yaw, dtype=np.float64)
    if width not in WIDTHS:
        raise ValueError("Supported widths are 5, 10 and 15 degrees.")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("yaw must be a finite vector.")
    if np.any((values < 0) | (values > 180)):
        raise ValueError("yaw must lie in [0,180] degrees.")
    return np.minimum((values // width).astype(np.int64), 180 // width - 1)


def load_dataset(
    cfg: dict, path: Path, override: Path | None = None
) -> BUAAInternalDataset:
    cache = (
        override.expanduser().resolve()
        if override
        else relative_path(path, cfg["data"]["cache_dir"])
    )
    dataset = BUAAInternalDataset(cache)
    metadata = dataset.cache_metadata
    if metadata["source_signature"] != cfg["data"]["expected_source_signature"]:
        raise ValueError("Dataset source differs from registered BUAA internal.")
    if metadata["normalization"] != "per_record_l2":
        raise ValueError("Expected per-record L2 signals.")
    if (
        dataset.input_size != cfg["model"]["input_size"]
        or dataset.num_classes != cfg["model"]["num_classes"]
    ):
        raise ValueError("Dataset shape/classes differ from setting.")
    for name in ("yaw", "pitch"):
        values = getattr(dataset, name)
        if values.shape != dataset.targets.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid or unaligned {name} metadata.")
    yaw_bins(dataset.yaw, cfg["split"]["bin_width_degrees"])
    return dataset


def make_splits(dataset: BUAAInternalDataset, cfg: dict) -> dict[str, np.ndarray]:
    bins = yaw_bins(dataset.yaw, cfg["split"]["bin_width_degrees"])
    pool = np.flatnonzero(bins % 2 == 0)
    test = np.flatnonzero(bins % 2 == 1)
    tr, va = stratified_train_test_split(dataset.targets[pool], 0.9, 1056)
    splits = {"train": pool[tr], "validation": pool[va], "test": test}
    validate_splits(splits, dataset.targets)
    if split_hash(splits) != cfg["split"]["expected_hash"]:
        raise ValueError("Regenerated split differs from registered indices.")
    return splits


def load_splits(
    dataset: BUAAInternalDataset, cfg: dict, path: Path
) -> dict[str, np.ndarray]:
    file = relative_path(path, cfg["split"]["directory"]) / "split_indices.npz"
    if sha256(file) != cfg["split"]["file_sha256"]:
        raise ValueError("Frozen split file SHA256 changed.")
    with np.load(file, allow_pickle=False) as saved:
        splits = {k: saved[k] for k in ("train", "validation", "test")}
    validate_splits(splits, dataset.targets)
    regenerated = make_splits(dataset, cfg)
    for key in splits:
        np.testing.assert_array_equal(splits[key], regenerated[key])
    return splits


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_frozen_split(path: Path, splits: dict[str, np.ndarray]) -> None:
    """Stable uncompressed NPZ: fixed header, byte order, member order and metadata."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ("train", "validation", "test"):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.ascontiguousarray(splits[name], dtype="<i8"),
                version=(1, 0),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def audit_summary(
    dataset: BUAAInternalDataset, splits: dict[str, np.ndarray], cfg: dict
) -> dict:
    width = cfg["split"]["bin_width_degrees"]
    return {
        "protocol": cfg["protocol"],
        "source_signature": dataset.cache_metadata["source_signature"],
        "bin_width_degrees": width,
        "split_hash": split_hash(splits),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "train_pool_intervals": [[x, x + width] for x in range(0, 180, 2 * width)],
        "test_intervals": [[x, x + width] for x in range(width, 180, 2 * width)],
        "interval_convention": "[left,right); final interval includes 180",
        "validation": "class-stratified 10% of even-bin pool, seed1056",
        "bin_count_ratio": "1:1; sample counts are not forced equal",
        "angle_buffer_degrees": 0,
        "file_disjoint": False,
        "shared_train_test_files": len(
            np.intersect1d(
                dataset.file_indices[splits["train"]],
                dataset.file_indices[splits["test"]],
            )
        ),
        "pitch_held_out": False,
        "conditioning_independent_split": True,
    }


def prepare(cfg: dict, path: Path, dataset: BUAAInternalDataset) -> dict:
    splits = make_splits(dataset, cfg)
    output = relative_path(path, cfg["split"]["directory"])
    summary = audit_summary(dataset, splits, cfg)
    if output.exists():
        load_splits(dataset, cfg, path)
        if json.loads((output / "summary.json").read_text()) != summary:
            raise ValueError("Existing split audit differs; refusing to overwrite.")
        return summary
    output.mkdir(parents=True, exist_ok=False)
    save_frozen_split(output / "split_indices.npz", splits)
    if sha256(output / "split_indices.npz") != cfg["split"]["file_sha256"]:
        raise ValueError("Generated NPZ file SHA256 differs from registered file.")
    write_json(output / "summary.json", summary)
    bins = yaw_bins(dataset.yaw, cfg["split"]["bin_width_degrees"])
    classes, boxes = [], []
    width = cfg["split"]["bin_width_degrees"]
    for key, ids in splits.items():
        for label, name in sorted(dataset.class_names.items()):
            selected = ids[dataset.targets[ids] == label]
            classes.append(
                {
                    "split": key,
                    "class_index": label,
                    "class_name": name,
                    "samples": len(selected),
                    "files": len(np.unique(dataset.file_indices[selected])),
                }
            )
            for box in range(180 // width):
                part = selected[bins[selected] == box]
                boxes.append(
                    {
                        "split": key,
                        "class_index": label,
                        "class_name": name,
                        "bin_index": box,
                        "left_deg": box * width,
                        "right_deg": (box + 1) * width,
                        "samples": len(part),
                    }
                )
    write_csv(output / "class_counts.csv", classes)
    write_csv(output / "class_yaw_counts.csv", boxes)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    cfg, path = read_setting(args.config)
    dataset = load_dataset(cfg, path, args.data_dir)
    print(json.dumps(prepare(cfg, path, dataset), indent=2))


if __name__ == "__main__":
    main()
