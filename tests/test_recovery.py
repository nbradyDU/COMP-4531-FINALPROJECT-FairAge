"""Tests CPU training continuation.

Purpose:
    Keep restarted training equivalent to uninterrupted training, including weights and
    history.

Tests:
    test_exact_cpu_continuation:
        Detects resumed CPU training diverging from uninterrupted training.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from fairage.train import status, train


def synthetic_loaders(config):
    """Provide four RGB images with independent, reproducible loader generators."""

    images = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(97))
    dataset = TensorDataset(images, torch.tensor([0, 2, 5, 8]))
    return tuple(DataLoader(dataset, batch_size=2, shuffle=index == 0,
                            generator=torch.Generator().manual_seed(4531 + index))
                 for index in range(2))


class SyntheticRecoveryChecks(unittest.TestCase):

    def test_exact_cpu_continuation(self):
        torch.set_num_threads(2)
        config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
        config.update(device="cpu", precision="fp32", experiment="scratch", epochs=2,
                      batch_size=2, workers=0, dropout=0.1, warmup_epochs=0,
                      patience=0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config.update(run_root=str(root / "runs"), results_root=str(root / "results"))
            with patch("fairage.train.make_loaders", side_effect=synthetic_loaders), \
                    contextlib.redirect_stdout(io.StringIO()):
                train(config, "continuous")
                stopped = train(config, "resumed", on_epoch=lambda epoch, row: False)
                self.assertEqual(stopped["epoch"], 1)
                train(config, "resumed", resume=True)
            directories = [root / "runs/baselines/scratch" / run for run in ("continuous", "resumed")]
            uninterrupted, resumed = [torch.load(directory / "latest.pt", weights_only=True, map_location="cpu")
                                      for directory in directories]
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(uninterrupted["scheduler"], resumed["scheduler"])
            self.assertEqual(uninterrupted["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
            for name, tensor in uninterrupted["model"].items():
                self.assertTrue(torch.equal(tensor, resumed["model"][name]), name)
            self.assertEqual(set(uninterrupted["optimizer"]["state"]), set(resumed["optimizer"]["state"]))
            for index, state in uninterrupted["optimizer"]["state"].items():
                for name, tensor in state.items():
                    self.assertTrue(torch.equal(tensor, resumed["optimizer"]["state"][index][name]), name)
            self.assertEqual(uninterrupted["rng"]["python"], resumed["rng"]["python"])
            self.assertEqual(uninterrupted["rng"]["numpy"], resumed["rng"]["numpy"])
            self.assertTrue(torch.equal(uninterrupted["rng"]["torch"], resumed["rng"]["torch"]))
            for left, right in zip(uninterrupted["rng"]["loaders"], resumed["rng"]["loaders"]):
                self.assertTrue(torch.equal(left, right))
            histories = [pd.read_csv(directory / "history.csv") for directory in directories]
            columns = [name for name in histories[0].columns if not name.endswith("seconds")]
            pd.testing.assert_frame_equal(histories[0][columns], histories[1][columns], check_exact=True)
            self.assertTrue((directories[1] / "best.pt").is_file())
            self.assertEqual(status(config, "resumed")["epoch"], 2)
            self.assertFalse(list(root.rglob("evaluation")))
