from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from TCN.data import smoke_splits, split_hash
from TCN.model import TCNClassifier
from TCN.utils import sha256
from TCN.angles import evaluation, train
from TCN.angles.backend import configure_backend
from TCN.angles.common import AngleBatches, load_checkpoint, restore_model, state_hash
from TCN.angles.model import AngleTCN
from TCN.angles.split import (
    ROOT,
    load_dataset,
    load_splits,
    read_setting,
    save_frozen_split,
    yaw_bins,
)


class AngleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.override = (
            Path(os.environ["TCN_DATA_DIR"]) if "TCN_DATA_DIR" in os.environ else None
        )
        cls.cfg, cls.path = read_setting(ROOT / "configs/yaw5.json")
        cls.dataset = load_dataset(cls.cfg, cls.path, cls.override)
        cls.splits = load_splits(cls.dataset, cls.cfg, cls.path)

    def test_boundaries_and_reject_invalid_angles(self) -> None:
        for width in (5, 10, 15):
            angles = np.array(
                [0, width - 1e-6, width, 2 * width - 1e-6, 2 * width, 180]
            )
            np.testing.assert_array_equal(
                yaw_bins(angles, width), [0, 0, 1, 1, 2, 180 // width - 1]
            )
            for bad in ([-1.0], [180.001], [np.nan], [[10.0]]):
                with self.assertRaises(ValueError):
                    yaw_bins(np.asarray(bad), width)
        with self.assertRaises(ValueError):
            yaw_bins(np.array([10.0]), 6)

    def test_three_frozen_splits_have_exact_holdout_and_stable_files(self) -> None:
        sizes = {
            5: (115533, 12842, 130028),
            10: (116063, 12902, 129438),
            15: (112674, 12523, 133206),
        }
        for width in (5, 10, 15):
            cfg, path = read_setting(ROOT / f"configs/yaw{width}.json")
            splits = load_splits(self.dataset, cfg, path)
            self.assertEqual(tuple(len(v) for v in splits.values()), sizes[width])
            bins = yaw_bins(self.dataset.yaw, width)
            for key in ("train", "validation"):
                self.assertTrue(np.all(bins[splits[key]] % 2 == 0))
            np.testing.assert_array_equal(splits["test"], np.flatnonzero(bins % 2 == 1))
            joined = np.concatenate(list(splits.values()))
            np.testing.assert_array_equal(np.sort(joined), np.arange(len(self.dataset)))
            for indices in splits.values():
                self.assertEqual(len(np.unique(self.dataset.targets[indices])), 10)
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / "split.npz"
                save_frozen_split(out, splits)
                self.assertEqual(sha256(out), cfg["split"]["file_sha256"])
            corrupt = {**cfg, "split": {**cfg["split"], "file_sha256": "0" * 64}}
            with self.assertRaises(ValueError):
                load_splits(self.dataset, corrupt, path)

    def test_identity_initialization_same_trunk_and_conditioning_gradients(
        self,
    ) -> None:
        torch.manual_seed(56)
        original = TCNClassifier().eval()
        values = torch.randn(2, 1, 800)
        counts = {"none": 144178, "yaw": 157426, "yaw_pitch": 157810, "pitch": 157426}
        for mode, count in counts.items():
            torch.manual_seed(56)
            model = AngleTCN(conditioning=mode).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertEqual(state_hash(model, True), state_hash(original, True))
            angles = (
                None
                if mode == "none"
                else torch.tensor([[10.0, 2.0], [120.0, 20.0]])[
                    :, : 2 if mode == "yaw_pitch" else 1
                ]
            )
            with torch.no_grad():
                torch.testing.assert_close(
                    model(values, angles), original(values), rtol=0, atol=0
                )
            if mode != "none":
                with self.assertRaises(ValueError):
                    model(values, None)
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    torch.nn.functional.cross_entropy(
                        model(values, angles), torch.tensor([0, 1])
                    ).backward()
                    self.assertTrue(
                        all(
                            p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()
                        )
                    )
                    optimizer.step()
                self.assertGreater(float(model.films[0][0].weight.grad.abs().sum()), 0)
                model.eval()
                changed = angles.clone()
                changed[:, -1] += 30
                with torch.no_grad():
                    self.assertFalse(
                        torch.allclose(
                            model(values, angles),
                            model(values, changed),
                            rtol=1e-7,
                            atol=1e-7,
                        )
                    )

    def test_batches_keep_signal_angle_label_alignment_and_training_coverage(
        self,
    ) -> None:
        selected = smoke_splits(self.splits, self.dataset.targets)
        for mode in ("none", "yaw", "yaw_pitch", "pitch"):
            batches = AngleBatches(self.dataset, selected, torch.device("cpu"), 7, mode)
            targets, conditions = [], []
            for values, angles, target in batches.batches("test"):
                targets.append(target.numpy())
                if angles is not None:
                    conditions.append(angles.numpy())
            np.testing.assert_array_equal(
                np.concatenate(targets), self.dataset.targets[selected["test"]]
            )
            if mode != "none":
                observed = np.concatenate(conditions)
                np.testing.assert_array_equal(
                    observed[:, 0],
                    (self.dataset.pitch if mode == "pitch" else self.dataset.yaw)[
                        selected["test"]
                    ],
                )
                if mode == "yaw_pitch":
                    np.testing.assert_array_equal(
                        observed[:, 1], self.dataset.pitch[selected["test"]]
                    )
            generator = torch.Generator().manual_seed(56)
            count = sum(
                len(target) for _, _, target in batches.batches("train", generator)
            )
            self.assertEqual(count, len(selected["train"]))
            with self.assertRaises(ValueError):
                list(batches.batches("test", generator))

    def test_train_selects_validation_then_tests_and_independent_eval_checks_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "train"
            command = [
                "--config",
                str(self.path),
                "--conditioning",
                "yaw_pitch",
                "--smoke",
                "--device",
                "cpu",
                "--cudnn-mode",
                "fast",
                "--threads",
                "2",
                "--output",
                str(output),
            ]
            if self.override:
                command += ["--data-dir", str(self.override)]
            args = train.make_parser().parse_args(command)
            calls = []
            original_evaluate = train.evaluate

            def tracked(model, data, split, names):
                history = output / "history.json"
                calls.append(
                    (
                        split,
                        len(json.loads(history.read_text())) if history.exists() else 0,
                    )
                )
                return original_evaluate(model, data, split, names)

            with patch.object(train, "evaluate", side_effect=tracked):
                result = train.run(args)
            self.assertEqual(
                calls, [("validation", 0), ("validation", 1), ("test", 2), ("test", 2)]
            )
            history = json.loads((output / "history.json").read_text())
            self.assertEqual(
                result["best_epoch"],
                min(history, key=lambda x: x["validation"]["loss"])["epoch"],
            )
            self.assertFalse(result["test_used_for_selection"])
            self.assertTrue(
                all(
                    x["training_samples_seen"] == 20 and x["updates"] == 3
                    for x in history
                )
            )
            saved = load_checkpoint(output / "best.pt")
            self.assertFalse(torch.backends.cudnn.deterministic)
            self.assertEqual(saved["provenance"]["cudnn_mode"], "fast")
            self.assertFalse(saved["provenance"]["backend"]["cudnn_deterministic"])
            with self.assertRaises(FileExistsError):
                train.run(args)
            bad = {
                **saved,
                "provenance": {**saved["provenance"], "bin_width_degrees": 15},
            }
            with self.assertRaises(ValueError):
                restore_model(
                    bad, self.dataset, self.cfg, self.splits, torch.device("cpu")
                )
            for split in ("validation", "test"):
                import argparse

                report = evaluation.run(
                    argparse.Namespace(
                        config=self.path,
                        checkpoint=output / "best.pt",
                        output=Path(temp) / split,
                        data_dir=self.override,
                        threads=2,
                        batch_size=8,
                        device="cpu",
                        smoke=False,
                    ),
                    split,
                )
                self.assertEqual(report["phase"], "smoke")
                self.assertEqual(report["metrics"]["samples"], 20)
                self.assertEqual(sum(row["samples"] for row in report["yaw_bins"]), 20)
                self.assertEqual(report["split_hash"], split_hash(self.splits))

    def test_compact_storage_preserves_training_weights_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            outputs = {}
            for compact in (False, True):
                output = Path(temp) / ("compact" if compact else "full")
                command = [
                    "--config",
                    str(self.path),
                    "--conditioning",
                    "yaw_pitch",
                    "--smoke",
                    "--device",
                    "cpu",
                    "--threads",
                    "2",
                    "--output",
                    str(output),
                ]
                if self.override:
                    command += ["--data-dir", str(self.override)]
                if compact:
                    command.append("--compact")
                outputs[compact] = (
                    output,
                    train.run(train.make_parser().parse_args(command)),
                )
            full, a = outputs[False]
            small, b = outputs[True]
            self.assertEqual(a["best_epoch"], b["best_epoch"])
            for name in ("best_test", "last_test"):
                self.assertEqual(a[name], b[name])
            original = load_checkpoint(full / "best.pt")
            narrowed = load_checkpoint(small / "best.pt")
            self.assertNotIn("optimizer_state_dict", narrowed)
            for name, value in original["model_state_dict"].items():
                torch.testing.assert_close(
                    value, narrowed["model_state_dict"][name], rtol=0, atol=0
                )
            self.assertFalse((small / "last.pt").exists())
            self.assertFalse((small / "last_test_predictions.npz").exists())
            with np.load(full / "best_test_predictions.npz") as x, np.load(
                small / "best_test_predictions.npz"
            ) as y:
                self.assertEqual(set(y.files), {"indices", "labels", "predictions"})
                for name in y.files:
                    np.testing.assert_array_equal(x[name], y[name])
                self.assertEqual(y["indices"].dtype, np.dtype("uint32"))
                self.assertEqual(y["predictions"].dtype, np.dtype("uint8"))
            size = lambda directory: sum(
                p.stat().st_size for p in directory.iterdir() if p.is_file()
            )
            self.assertLess(size(small), size(full) / 2)
            self.assertNotIn(
                "per_class",
                json.loads((small / "history.json").read_text())[0]["validation"],
            )
            restore_model(
                narrowed, self.dataset, self.cfg, self.splits, torch.device("cpu")
            )

    def test_backend_policy_and_restored_checkpoint_match(self) -> None:
        original = (
            torch.backends.cudnn.enabled,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )
        try:
            strict = configure_backend("strict")
            self.assertTrue(strict["cudnn_deterministic"])
            fast = configure_backend("fast")
            self.assertFalse(fast["cudnn_deterministic"])
            self.assertFalse(fast["cudnn_benchmark"])
            model = AngleTCN(conditioning="yaw")
            saved = {
                "model_state_dict": model.state_dict(),
                "provenance": {
                    "protocol": self.cfg["protocol"],
                    "bin_width_degrees": 5,
                    "source_signature": self.dataset.cache_metadata["source_signature"],
                    "split_hash": split_hash(self.splits),
                    "class_names": self.dataset.class_names,
                    "model": model.description(),
                    "conditioning": "yaw",
                    "cudnn_mode": "fast",
                },
            }
            configure_backend("strict")
            restore_model(
                saved, self.dataset, self.cfg, self.splits, torch.device("cpu")
            )
            self.assertFalse(torch.backends.cudnn.deterministic)
            with self.assertRaises(ValueError):
                configure_backend("invalid")
        finally:
            (
                torch.backends.cudnn.enabled,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
            ) = original


if __name__ == "__main__":
    unittest.main()
