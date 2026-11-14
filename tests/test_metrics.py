"""Tests classification metrics, subgroup tables, and image preprocessing.

Purpose:
    Keep reported scores, subgroup counts, and model inputs consistent with the
    experiment.

Tests:
    test_empty_group_metrics:
        Detects empty groups being reported as measured scores instead of missing
        values.

    test_class_coverage_and_ordinal_errors:
        Catches incorrect class averaging or age-bin error scores on known label
        examples.

    test_confidence_metrics_distinguish_bin_counts:
        Checks ECE against independently calculated 10-bin and 15-bin values.

    test_confidence_bin_edges_and_strict_threshold:
        Checks internal bin edges, confidence 1, and the strict 90% error threshold.

    test_evaluation_preprocessing:
        Catches changes to training crop/flip settings or normalized evaluation images
        and labels.

    test_subgroup_tables:
        Catches incorrect confusion tables, subgroup counts, or missing values for
        unobserved groups.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from fairage.data import AGE_CLASSES, RACE_GROUPS, FairFaceDataset
from fairage.evaluate import save_analysis
from fairage.metrics import classification_metrics, confidence_metrics, scientific_metrics


class ScientificChecks(unittest.TestCase):

    def test_empty_group_metrics(self):
        expected = {"accuracy": None, "macro_f1": None, "balanced_accuracy": None,
                    "within_one_bin_accuracy": None, "mean_absolute_bin_index_error": None}
        self.assertEqual(scientific_metrics([], []), expected)

    def test_class_coverage_and_ordinal_errors(self):
        cases = [
            ([0, 0, 1, 1, 1, 2], [0, 1, 1, 2, 2, 2], (.5, 47 / 270, 11 / 18, 1., .5)),
            (list(range(9)), list(range(9)), (1., 1., 1., 1., 0.)),
            ([0, 0], [0, 0], (1., 1 / 9, 1., 1., 0.)),
            ([0, 1, 2], [0, 2, 1], (1 / 3, 1 / 9, 1 / 3, 1., 2 / 3)),
            ([0, 0, 8], [0, 8, 8], (2 / 3, 4 / 27, .75, 2 / 3, 8 / 3)),
            ([0, 1], [1, 0], (0., 0., 0., 1., 1.)),
        ]
        keys = ("accuracy", "macro_f1", "balanced_accuracy",
                "within_one_bin_accuracy", "mean_absolute_bin_index_error")
        for actual, predicted, expected in cases:
            with self.subTest(actual=actual, predicted=predicted):
                scientific = scientific_metrics(actual, predicted)
                for name, value in zip(keys, expected):
                    self.assertAlmostEqual(scientific[name], value)
                self.assertEqual(classification_metrics(actual, predicted),
                                 {name: scientific[name] for name in keys[:2]})

    def test_confidence_metrics_distinguish_bin_counts(self):
        probabilities = np.array([[.61, *([.39 / 8] * 8)],
                                  [.68, *([.32 / 8] * 8)]])
        summary = confidence_metrics([0, 1], probabilities)
        self.assertAlmostEqual(summary["mean_max_probability"], .645)
        # Ten bins combine both rows: |(.61 + .68)/2 - 1/2| = .145.
        self.assertAlmostEqual(summary["ece_10"], .145)
        # Fifteen bins separate them: ((1 - .61) + .68)/2 = .535.
        self.assertAlmostEqual(summary["ece_15"], .535)
        self.assertEqual((summary["wrong_count"], summary["wrong_above_90"]), (1, 0))

    def test_confidence_bin_edges_and_strict_threshold(self):
        probabilities = np.array([[p, *([(1 - p) / 8] * 8)] for p in (.59, .6, .9, 1.)])
        summary = confidence_metrics([1, 0, 1, 1], probabilities)
        self.assertAlmostEqual(summary["mean_max_probability"], .7725)
        # .59 and .6 belong to different bins; 1 belongs to the final bin.
        # Weighted absolute gaps: (.59 + (1 - .6) + .9 + 1)/4 = .7225.
        self.assertAlmostEqual(summary["ece_10"], .7225)
        self.assertAlmostEqual(summary["ece_15"], .7225)
        self.assertEqual(summary["wrong_count"], 3)
        self.assertEqual(summary["wrong_above_90"], 1)  # Exactly .9 is excluded.

    def test_evaluation_preprocessing(self):
        rows = pd.DataFrame({"image_path": ["image.png"], "target": [0]})
        with tempfile.TemporaryDirectory() as temporary, patch("fairage.data.read_manifest", return_value=rows):
            Image.new("RGB", (260, 260), (255, 0, 0)).save(Path(temporary) / "image.png")
            training = FairFaceDataset(temporary, "train", "crop")
            self.assertIsInstance(training.transform.transforms[1], transforms.RandomResizedCrop)
            self.assertEqual(training.transform.transforms[2].p, .5)
            for split in ("train", "development", "holdout"):
                dataset = FairFaceDataset(temporary, split, "crop", augment=False)
                first, target = dataset[0]
                self.assertTrue(torch.equal(first, dataset[0][0]))
                self.assertEqual(target, 0)
                expected = torch.tensor([(1-.485)/.229, -.456/.224, -.406/.225])
                torch.testing.assert_close(first, expected[:, None, None].expand(3, 224, 224))

    def test_subgroup_tables(self):
        records = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2],
                                "predicted": [0, 1, 1, 2, 2, 2],
                                "race": [RACE_GROUPS[0]] * 6})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_analysis(records, output)
            counts = pd.read_csv(output / "confusion.csv", index_col=0)
            np.testing.assert_array_equal(counts.iloc[:3, :3], [[1, 1, 0], [0, 1, 2], [0, 0, 1]])
            normalized = pd.read_csv(output / "confusion_normalized.csv", index_col=0)
            np.testing.assert_allclose(normalized.iloc[:3, :3], [[.5, .5, 0], [0, 1/3, 2/3], [0, 0, 1]])
            self.assertTrue((normalized.iloc[3:] == 0).all().all())
            age = pd.read_csv(output / "by_age.csv")
            self.assertEqual(age.age.tolist(), AGE_CLASSES)
            self.assertEqual(age.support.tolist(), [2, 3, 1, 0, 0, 0, 0, 0, 0])
            race = pd.read_csv(output / "by_race.csv")
            self.assertEqual(race.support.tolist(), [6, 0, 0, 0, 0, 0, 0])
            self.assertAlmostEqual(race.accuracy.iloc[0], .5)
            self.assertAlmostEqual(race.macro_f1.iloc[0], 47 / 270)
            self.assertTrue(race.loc[race.support == 0, "accuracy"].isna().all())
            cells = pd.read_csv(output / "by_age_race.csv")
            self.assertEqual(len(cells), 63)
            self.assertEqual(cells.support.sum(), 6)
