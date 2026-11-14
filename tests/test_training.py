"""Tests checkpoint selection, evaluation, result comparisons, and run locking.

Purpose:
    Prevent selection of the wrong epoch, mismatched result comparisons, or two writers
    changing the same run.

Tests:
    test_earliest_best_checkpoint_survives_tie_and_decline:
        Detects later tied or worse development scores replacing the earliest best
        checkpoint.

    test_checkpoint_evaluation_and_matched_comparisons:
        Detects incorrect saved scores or comparisons of different records, labels,
        checkpoints, or precision.

    test_run_lock_rejects_a_second_writer:
        Detects a second process resuming a run while another writer holds its lock.
"""

import contextlib
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import yaml
from filelock import FileLock
from torch.utils.data import TensorDataset

from fairage.analysis import load_results
from fairage.data import AGE_CLASSES, RACE_GROUPS
from fairage.evaluate import evaluate_checkpoint
from fairage.train import status, train
from test_recovery import synthetic_loaders


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
        self.config.update(device="cpu", precision="fp32", experiment="scratch",
                           epochs=4, batch_size=2, workers=0)

    def test_earliest_best_checkpoint_survives_tie_and_decline(self):
        model = torch.nn.Linear(1, 9)
        scores = [.2, .4, .4, .3]

        def advance(model, loader, optimizer, config):
            with torch.no_grad():
                model.weight.add_(1)
            optimizer.step()
            return {"loss": 1., "accuracy": .5, "macro_f1": .2, "seconds": 0.}

        with tempfile.TemporaryDirectory() as temporary:
            self.config["run_root"] = temporary
            with patch("fairage.train.configure_model", return_value=model), \
                    patch("fairage.train.make_loaders", side_effect=synthetic_loaders), \
                    patch("fairage.train.train_one_epoch", side_effect=advance), \
                    patch("fairage.train.evaluate_loader", side_effect=[
                        ({"loss": 1., "accuracy": .5, "macro_f1": f1, "seconds": 0.}, None) for f1 in scores]), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = train(self.config, "selection")
            directory = Path(temporary) / "baselines/scratch/selection"
            best, latest = [torch.load(directory / f"{name}.pt", weights_only=True, map_location="cpu")
                            for name in ("best", "latest")]
            self.assertEqual((result["best_epoch"], best["epoch"], latest["epoch"]), (2, 2, 4))
            self.assertEqual(best["best_score"], .4)
            torch.testing.assert_close(latest["model"]["weight"] - best["model"]["weight"], torch.full_like(model.weight, 2.))
            self.assertEqual([row["dev_macro_f1"] for row in latest["history"]], scores)

    def test_checkpoint_evaluation_and_matched_comparisons(self):
        targets, predicted = torch.tensor([0, 0, 1, 1, 1, 2]), torch.tensor([0, 1, 1, 2, 2, 2])
        logits = torch.zeros(6, 9)
        logits[torch.arange(6), predicted] = math.log(8)
        dataset = TensorDataset(logits, targets)
        dataset.rows = pd.DataFrame({"sample_id": [f"sample-{i}" for i in range(6)], "target": targets.numpy(),
                                     "age": [AGE_CLASSES[i] for i in targets], "race": [RACE_GROUPS[0]] * 6})
        model = torch.nn.Linear(9, 9, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(9))
        with tempfile.TemporaryDirectory() as temporary:
            self.config["run_root"] = temporary
            directory = Path(temporary) / "baselines/scratch/evaluation"
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(json.dumps(self.config))
            pd.DataFrame({"epoch": [1, 2, 3], "dev_macro_f1": [.2, .4, .3]}).to_csv(directory / "history.csv", index=False)
            torch.save({"format_version": 1, "config": self.config, "epoch": 2,
                        "model": model.state_dict()}, directory / "best.pt")
            with patch("fairage.evaluate.FairFaceDataset", return_value=dataset) as inputs, \
                    patch("fairage.evaluate.DeiTTiny", return_value=model), \
                    patch("fairage.evaluate.save_evaluation_figures"):
                result = evaluate_checkpoint(self.config, "evaluation", device="cpu")
            inputs.assert_called_once_with(self.config["data_root"], "development", limit=None, augment=False)
            output = Path(result["output"])
            summary = json.loads((output / "metrics.json").read_text())
            self.assertEqual((summary["epoch"], summary["precision"], summary["count"]), (2, "fp32", 6))
            self.assertAlmostEqual(summary["loss"], 2.5 * math.log(2), places=6)
            self.assertAlmostEqual(summary["mean_max_probability"], .5, places=6)
            self.assertAlmostEqual(summary["ece_10"], 0., places=6)
            self.assertAlmostEqual(summary["ece_15"], 0., places=6)
            self.assertEqual((summary["wrong_count"], summary["wrong_above_90"]), (3, 0))
            for key, value in {"accuracy": .5, "macro_f1": 47/270, "balanced_accuracy": 11/18,
                               "within_one_bin_accuracy": 1., "mean_absolute_bin_index_error": .5}.items():
                self.assertAlmostEqual(summary[key], value)
            records = pd.read_csv(output / "predictions.csv")
            self.assertEqual(records.predicted.tolist(), predicted.tolist())
            self.assertEqual(records.target.tolist(), targets.tolist())
            other = directory.with_name("other")
            shutil.copytree(directory, other)
            other_output = other / "evaluation" / output.name
            other_summary = dict(summary, run="other")
            (other_output / "metrics.json").write_text(json.dumps(other_summary))
            self.assertEqual(load_results(self.config, ["evaluation", "other"])["metrics"].epoch.tolist(), [2, 2])
            for field in ("sample_id", "target", "race", "precision", "split", "checkpoint", "epoch"):
                with self.subTest(field=field):
                    changed, identity = other_summary.copy(), records.copy()
                    if field in identity:
                        identity.loc[0, field] = 1 if field == "target" else "different"
                    else:
                        changed[field] = 99 if field == "epoch" else "different"
                    identity.to_csv(other_output / "predictions.csv", index=False)
                    (other_output / "metrics.json").write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, "identical|identity"):
                        load_results(self.config, ["evaluation", "other"])

    def test_run_lock_rejects_a_second_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.config["run_root"] = temporary
            directory = Path(temporary) / "baselines/scratch/locked"
            directory.mkdir(parents=True)
            config_path = Path(temporary) / "config.yaml"
            config_path.write_text(yaml.safe_dump(self.config))
            with FileLock(directory / ".lock"):
                self.assertTrue(status(self.config, "locked")["active"])
                result = subprocess.run([sys.executable, "-B", "-m", "fairage", "resume", "locked",
                                         "--config", str(config_path)], capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("could not be acquired", result.stderr)
