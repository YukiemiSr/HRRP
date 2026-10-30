"""BUAA binary HRRP loading, caching, and reproducible Setting A splitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]
HRRP_BINS = 800
RECORD_DTYPE = np.dtype(
    [
        ("start_flag1", "<u4"),
        ("start_flag2", "<u4"),
        ("time1", "<u4"),
        ("time2", "<u4"),
        ("target_id", "<u4"),
        ("yaw", "<f8"),
        ("pitch", "<f8"),
        ("distance", "<f8"),
        ("azimuth", "<f8"),
        ("course", "<f8"),
        ("label", "<i4"),
        ("hrrp", "<f8", (HRRP_BINS,)),
        ("end_flag1", "<u4"),
        ("end_flag2", "<u4"),
    ]
)
EXPECTED_FLAGS = (0x55AAAA55, 0xAA5555AA, 0xBB6666BB, 0x66BBBBAA)
CACHE_VERSION = 1


def _source_signature(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare_internal_cache(
    data_dir: PathLike,
    label_json: PathLike,
    cache_dir: PathLike,
) -> Mapping[str, object]:
    """Create an L2-normalized, memory-mapped cache; reuse it when valid."""
    source = Path(data_dir).resolve()
    label_path = Path(label_json).resolve()
    cache = Path(cache_dir).resolve()
    paths = sorted(source.glob("*.dat"))
    if not paths:
        raise FileNotFoundError(f"No .dat files found in {source}.")
    labels_by_raw = json.loads(label_path.read_text(encoding="utf-8"))
    signature = _source_signature(paths)
    metadata_path = cache / "cache_metadata.json"
    required = [
        cache / "signals.npy",
        cache / "raw_labels.npy",
        cache / "yaw.npy",
        cache / "pitch.npy",
        cache / "azimuth.npy",
        cache / "course.npy",
        cache / "target_ids.npy",
        cache / "file_indices.npy",
        cache / "row_indices.npy",
        cache / "source_files.json",
    ]
    if metadata_path.is_file() and all(path.is_file() for path in required):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("cache_version") == CACHE_VERSION
            and metadata.get("source_signature") == signature
            and metadata.get("labels_by_raw") == labels_by_raw
        ):
            return metadata

    cache.mkdir(parents=True, exist_ok=True)
    counts = []
    for path in paths:
        size = path.stat().st_size
        if size % RECORD_DTYPE.itemsize:
            raise ValueError(
                f"{path} is truncated: {size} is not divisible by "
                f"{RECORD_DTYPE.itemsize}."
            )
        counts.append(size // RECORD_DTYPE.itemsize)
    total = int(sum(counts))
    arrays = {
        "signals": np.lib.format.open_memmap(
            cache / "signals.npy", mode="w+", dtype=np.float32, shape=(total, HRRP_BINS)
        ),
        "raw_labels": np.lib.format.open_memmap(
            cache / "raw_labels.npy", mode="w+", dtype=np.int32, shape=(total,)
        ),
        "yaw": np.lib.format.open_memmap(
            cache / "yaw.npy", mode="w+", dtype=np.float32, shape=(total,)
        ),
        "pitch": np.lib.format.open_memmap(
            cache / "pitch.npy", mode="w+", dtype=np.float32, shape=(total,)
        ),
        "azimuth": np.lib.format.open_memmap(
            cache / "azimuth.npy", mode="w+", dtype=np.float32, shape=(total,)
        ),
        "course": np.lib.format.open_memmap(
            cache / "course.npy", mode="w+", dtype=np.float32, shape=(total,)
        ),
        "target_ids": np.lib.format.open_memmap(
            cache / "target_ids.npy", mode="w+", dtype=np.uint32, shape=(total,)
        ),
        "file_indices": np.lib.format.open_memmap(
            cache / "file_indices.npy", mode="w+", dtype=np.int32, shape=(total,)
        ),
        "row_indices": np.lib.format.open_memmap(
            cache / "row_indices.npy", mode="w+", dtype=np.uint32, shape=(total,)
        ),
    }
    offset = 0
    class_counts: Dict[str, int] = {key: 0 for key in labels_by_raw}
    for file_index, (path, count) in enumerate(zip(paths, counts)):
        records = np.memmap(path, mode="r", dtype=RECORD_DTYPE)
        flags_ok = (
            (records["start_flag1"] == EXPECTED_FLAGS[0])
            & (records["start_flag2"] == EXPECTED_FLAGS[1])
            & (records["end_flag1"] == EXPECTED_FLAGS[2])
            & (records["end_flag2"] == EXPECTED_FLAGS[3])
        )
        if not bool(flags_ok.all()):
            raise ValueError(f"{path} contains {int((~flags_ok).sum())} invalid records.")
        stop = offset + count
        signals = np.asarray(records["hrrp"], dtype=np.float64)
        if not np.isfinite(signals).all():
            raise ValueError(f"{path} contains non-finite HRRP values.")
        norms = np.linalg.norm(signals, axis=1, keepdims=True)
        if bool((norms == 0).any()):
            raise ValueError(f"{path} contains all-zero HRRP records.")
        raw_labels = np.asarray(records["label"], dtype=np.int32)
        unknown_labels = sorted(set(map(str, np.unique(raw_labels))) - set(labels_by_raw))
        if unknown_labels:
            raise ValueError(f"{path} contains labels absent from {label_path}: {unknown_labels}.")
        arrays["signals"][offset:stop] = (signals / np.maximum(norms, 1e-12)).astype(
            np.float32
        )
        arrays["raw_labels"][offset:stop] = raw_labels
        for name in ("yaw", "pitch", "azimuth", "course"):
            values = np.asarray(records[name], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{path} contains non-finite {name} values.")
            arrays[name][offset:stop] = values
        arrays["target_ids"][offset:stop] = records["target_id"]
        arrays["file_indices"][offset:stop] = file_index
        arrays["row_indices"][offset:stop] = np.arange(count, dtype=np.uint32)
        for raw_label, label_count in zip(*np.unique(raw_labels, return_counts=True)):
            class_counts[str(int(raw_label))] += int(label_count)
        offset = stop
    for array in arrays.values():
        array.flush()
    _write_json(cache / "source_files.json", [path.name for path in paths])
    metadata = {
        "cache_version": CACHE_VERSION,
        "source_signature": signature,
        "source_directory": str(source),
        "records": total,
        "files": len(paths),
        "input_size": HRRP_BINS,
        "normalization": "per_record_l2",
        "labels_by_raw": labels_by_raw,
        "class_counts": class_counts,
    }
    _write_json(metadata_path, metadata)
    return metadata


class BUAAInternalDataset(Dataset):
    """Memory-mapped BUAA internal dataset with continuous training targets."""

    def __init__(self, cache_dir: PathLike) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        metadata_path = self.cache_dir / "cache_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"BUAA cache is missing: {metadata_path}.")
        self.cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.data = np.load(self.cache_dir / "signals.npy", mmap_mode="r")
        self.raw_labels = np.load(self.cache_dir / "raw_labels.npy", mmap_mode="r")
        self.yaw = np.load(self.cache_dir / "yaw.npy", mmap_mode="r")
        self.pitch = np.load(self.cache_dir / "pitch.npy", mmap_mode="r")
        self.azimuth = np.load(self.cache_dir / "azimuth.npy", mmap_mode="r")
        self.course = np.load(self.cache_dir / "course.npy", mmap_mode="r")
        self.target_ids = np.load(self.cache_dir / "target_ids.npy", mmap_mode="r")
        self.file_indices = np.load(self.cache_dir / "file_indices.npy", mmap_mode="r")
        self.row_indices = np.load(self.cache_dir / "row_indices.npy", mmap_mode="r")
        self.source_files = json.loads(
            (self.cache_dir / "source_files.json").read_text(encoding="utf-8")
        )
        labels_by_raw = self.cache_metadata["labels_by_raw"]
        self.raw_label_values = sorted(int(value) for value in labels_by_raw)
        self.label_to_index = {
            raw_label: index for index, raw_label in enumerate(self.raw_label_values)
        }
        self.class_names = {
            self.label_to_index[int(raw_label)]: class_name
            for raw_label, class_name in labels_by_raw.items()
        }
        self.num_classes = len(self.class_names)
        self.input_size = int(self.data.shape[1])
        self.targets = np.fromiter(
            (self.label_to_index[int(value)] for value in self.raw_labels),
            dtype=np.int64,
            count=len(self.raw_labels),
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = torch.from_numpy(np.array(self.data[index], copy=True)).unsqueeze(0)
        return signal, torch.tensor(int(self.targets[index]), dtype=torch.long)

    def metadata(self, index: int) -> Dict[str, object]:
        file_index = int(self.file_indices[index])
        class_index = int(self.targets[index])
        return {
            "sample_id": f"{self.source_files[file_index]}#row={int(self.row_indices[index])}",
            "source_file": self.source_files[file_index],
            "row_index": int(self.row_indices[index]),
            "target_id": int(self.target_ids[index]),
            "raw_label": int(self.raw_labels[index]),
            "class_index": class_index,
            "class_name": self.class_names[class_index],
            "yaw_deg": float(self.yaw[index]),
            "pitch_deg": float(self.pitch[index]),
            "azimuth_deg": float(self.azimuth[index]),
            "course_deg": float(self.course[index]),
        }


def stratified_train_test_split(
    targets: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split every class independently, then shuffle the two index sets."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between zero and one.")
    labels = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for class_index in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == class_index)
        shuffled = rng.permutation(indices)
        train_count = int(np.floor(len(indices) * train_ratio))
        if train_count <= 0 or train_count >= len(indices):
            raise ValueError(f"Class {class_index} is too small for the requested split.")
        train_parts.append(shuffled[:train_count])
        test_parts.append(shuffled[train_count:])
    train_indices = rng.permutation(np.concatenate(train_parts)).astype(np.int64)
    test_indices = rng.permutation(np.concatenate(test_parts)).astype(np.int64)
    if np.intersect1d(train_indices, test_indices).size:
        raise RuntimeError("Train/test split overlap detected.")
    if len(train_indices) + len(test_indices) != len(labels):
        raise RuntimeError("Train/test split does not cover the complete dataset.")
    return train_indices, test_indices
