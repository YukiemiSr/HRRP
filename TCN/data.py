"""Frozen historical 8:2 split, independent data loading, and resident batches."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

if __package__:
    from .dataset import BUAAInternalDataset, stratified_train_test_split
    from .utils import read_config, relative_path, sha256
else:
    from dataset import BUAAInternalDataset, stratified_train_test_split
    from utils import read_config, relative_path, sha256
import hashlib


def split_hash(splits: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("train", "validation", "test"):
        digest.update(key.encode())
        digest.update(splits[key].astype("<i8").tobytes())
    return digest.hexdigest()


def make_splits(targets: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    cfg = config["split"]
    seed = int(cfg["seed"])
    pool, test = stratified_train_test_split(targets, cfg["train_pool_ratio"], seed)
    tr, va = stratified_train_test_split(
        targets[pool], 1 - cfg["validation_fraction_of_train_pool"], seed + 1000
    )
    return {"train": pool[tr], "validation": pool[va], "test": test}


def validate_splits(splits: dict[str, np.ndarray], targets: np.ndarray) -> None:
    joined = np.concatenate([splits[k] for k in ("train", "validation", "test")])
    if joined.dtype.kind not in "iu" or not np.array_equal(
        np.sort(joined), np.arange(len(targets))
    ):
        raise ValueError(
            "Split must cover every record once, without overlap/out-of-range indices."
        )
    labels = np.unique(targets)
    for key in ("train", "validation", "test"):
        if splits[key].ndim != 1 or not np.array_equal(
            np.unique(targets[splits[key]]), labels
        ):
            raise ValueError(f"Missing class or invalid indices in {key}.")


def load_data(config: dict, config_path: Path, override: Path | None = None):
    cache = (
        override.resolve()
        if override
        else relative_path(config_path, config["data"]["cache_dir"])
    )
    dataset = BUAAInternalDataset(cache)
    if (
        dataset.cache_metadata["source_signature"]
        != config["data"]["expected_source_signature"]
    ):
        raise ValueError("BUAA source signature differs from the registered baseline.")
    if dataset.cache_metadata["normalization"] != "per_record_l2":
        raise ValueError("Expected per-record L2 cache.")
    if (
        dataset.input_size != config["model"]["input_size"]
        or dataset.num_classes != config["model"]["num_classes"]
    ):
        raise ValueError("Input size or number of classes differs from model config.")
    split_file = relative_path(config_path, config["split"]["path"])
    if sha256(split_file) != config["split"]["file_sha256"]:
        raise ValueError("Frozen split file hash changed.")
    with np.load(split_file) as saved:
        splits = {k: saved[k] for k in ("train", "validation", "test")}
    validate_splits(splits, dataset.targets)
    if split_hash(splits) != config["split"]["expected_hash"]:
        raise ValueError("Frozen split indices changed.")
    return dataset, splits


def smoke_splits(
    splits: dict[str, np.ndarray], targets: np.ndarray
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(56)
    result = {}
    for key, ids in splits.items():
        selected = [
            rng.choice(ids[targets[ids] == label], 2, replace=False)
            for label in np.unique(targets)
        ]
        result[key] = rng.permutation(np.concatenate(selected)).astype(np.int64)
    return result


class ResidentBatches:
    def __init__(
        self,
        dataset,
        splits: dict[str, np.ndarray],
        device: torch.device,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        chosen = np.unique(np.concatenate(list(splits.values())))
        self.signals = torch.tensor(
            np.array(dataset.data[chosen], copy=True), device=device
        ).unsqueeze(1)
        self.targets = torch.tensor(dataset.targets[chosen], device=device)
        self.indices = {
            k: torch.tensor(np.searchsorted(chosen, v), device=device)
            for k, v in splits.items()
        }
        self.batch_size = batch_size

    def batches(self, split: str, generator: torch.Generator | None = None):
        ids = self.indices[split]
        if generator is not None:
            if split != "train":
                raise ValueError("Only training may be shuffled.")
            order = torch.randperm(len(ids), generator=generator)
            ids = ids[order.to(ids.device)]
        for part in ids.split(self.batch_size):
            yield self.signals[part], self.targets[part]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--write-split",
        type=Path,
        help="Regenerate the historical split into a new NPZ file.",
    )
    args = parser.parse_args()
    cfg, path = read_config(args.config)
    dataset, splits = load_data(cfg, path, args.data_dir)
    regenerated = make_splits(dataset.targets, cfg)
    for k in splits:
        np.testing.assert_array_equal(regenerated[k], splits[k])
    if args.write_split:
        if args.write_split.exists():
            raise FileExistsError(args.write_split)
        args.write_split.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.write_split, **regenerated)
    print(
        {
            "records": len(dataset),
            "classes": dataset.num_classes,
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "split_hash": split_hash(splits),
        }
    )


if __name__ == "__main__":
    main()
