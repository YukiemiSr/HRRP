"""Meaningful checks for frozen data, original weights and standalone train/val/test."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from data import load_data, make_splits, split_hash, validate_splits
import train as train_module
from utils import classification_metrics, load_model, read_config


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.config, cls.path = read_config(PACKAGE / "config.json")
        cls.override = (
            Path(os.environ["TCN_DATA_DIR"]).resolve()
            if "TCN_DATA_DIR" in os.environ
            else None
        )
        cls.dataset, cls.splits = load_data(cls.config, cls.path, cls.override)

    def test_frozen_split_regenerates_and_invalid_overlap_is_rejected(self) -> None:
        self.assertEqual(
            {k: len(v) for k, v in self.splits.items()},
            {"train": 186041, "validation": 20677, "test": 51685},
        )
        regenerated = make_splits(self.dataset.targets, self.config)
        for k in self.splits:
            np.testing.assert_array_equal(regenerated[k], self.splits[k])
        broken = {k: v.copy() for k, v in self.splits.items()}
        broken["train"][0] = broken["test"][0]
        with self.assertRaises(ValueError):
            validate_splits(broken, self.dataset.targets)

    def test_original_checkpoint_loads_and_all_parameters_receive_gradients(
        self,
    ) -> None:
        model, saved = load_model(
            PACKAGE / "checkpoints/best.pt",
            self.dataset,
            split_hash(self.splits),
            torch.device("cpu"),
        )
        self.assertEqual(saved["epoch"], 99)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 144178)
        x = torch.tensor(
            np.array(self.dataset.data[self.splits["train"][:2]], copy=True)
        ).unsqueeze(1)
        y = torch.tensor(self.dataset.targets[self.splits["train"][:2]])
        torch.nn.functional.cross_entropy(model(x), y).backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
        with self.assertRaises(ValueError):
            model(torch.randn(2, 1, 16))
        with self.assertRaises(ValueError):
            load_model(
                PACKAGE / "checkpoints/best.pt",
                self.dataset,
                "wrong-split",
                torch.device("cpu"),
            )

    def test_training_selects_only_on_validation_and_tests_only_after_training(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            calls = []

            def tracked(model, data, split, names):
                h = out / "history.json"
                completed = len(json.loads(h.read_text())) if h.exists() else 0
                calls.append((split, completed))
                return original(model, data, split, names)

            original = train_module.evaluate
            args = argparse.Namespace(
                config=self.path,
                data_dir=self.override,
                device="cpu",
                batch_size=None,
                threads=2,
                seed=None,
                epochs=None,
                smoke=True,
                output=out,
            )
            with patch.object(train_module, "evaluate", side_effect=tracked):
                result = train_module.run(args)
            self.assertEqual(
                calls, [("validation", 0), ("validation", 1), ("test", 2), ("test", 2)]
            )
            history = json.loads((out / "history.json").read_text())
            self.assertEqual(
                result["best_epoch"],
                min(history, key=lambda r: r["validation"]["loss"])["epoch"],
            )
            self.assertEqual(result["phase"], "smoke")
            self.assertFalse(result["test_used_for_selection"])
            self.assertTrue(all(r["training_samples_seen"] == 20 for r in history))
            with np.load(out / "best_test_predictions.npz") as z:
                metrics = classification_metrics(
                    z["labels"], z["predictions"], self.dataset.class_names
                )
                self.assertEqual(metrics["accuracy"], result["best_test"]["accuracy"])
            with self.assertRaises(FileExistsError):
                train_module.run(args)

    def test_directory_runs_without_parent_project_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            portable = Path(temp) / "portable"
            portable.mkdir()
            for name in (
                "__init__.py",
                "model.py",
                "tcn.py",
                "data.py",
                "dataset.py",
                "utils.py",
                "train.py",
                "val.py",
                "test.py",
                "config.json",
            ):
                shutil.copy2(PACKAGE / name, portable / name)
            shutil.copytree(PACKAGE / "checkpoints", portable / "checkpoints")
            shutil.copytree(PACKAGE / "splits", portable / "splits")
            env = {**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"}
            data_dir = str(self.dataset.cache_dir)
            common = ["--data-dir", data_dir, "--device", "cpu", "--threads", "2"]
            completed = subprocess.run(
                [sys.executable, "train.py", *common, "--smoke"],
                cwd=portable,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            checkpoint = portable / "runs/smoke_seed_56/best.pt"
            for script, split in (("val.py", "validation"), ("test.py", "test")):
                output = portable / f"{split}.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        script,
                        *common,
                        "--checkpoint",
                        str(checkpoint),
                        "--smoke",
                        "--output",
                        str(output),
                    ],
                    cwd=portable,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )
                result = json.loads(output.read_text())
                self.assertEqual(result["split"], split)
                self.assertEqual(result["phase"], "smoke")
                self.assertEqual(result["metrics"]["samples"], 20)


if __name__ == "__main__":
    unittest.main()
