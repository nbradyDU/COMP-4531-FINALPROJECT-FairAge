"""Tests FairFace split preparation and manifest loading.

Purpose:
    Keep exact duplicate images out of training partitions and preserve the original
    split definitions.

Tests:
    test_original_exclusions_membership_serialization_and_loader:
        Detects duplicate leakage, changed seeded split membership, or unreadable saved
        manifests.

    test_existing_destination_refused_before_hashing:
        Detects overwriting an existing split destination or hashing inputs before
        refusing it.

    test_missing_image_creates_no_candidate:
        Detects incomplete split output left behind when a required image is missing.

    test_publisher_paths_cannot_escape_image_root:
        Detects publisher paths outside the image directory being hashed or accepted.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fairage.data import AGE_CLASSES, RACE_GROUPS, prepare_splits, read_manifest


def publisher_fixture(root):
    """Build nine ages, an internal duplicate/overlap, and a separate overlap."""

    raw = root / "raw/fairface-padding-0.25"
    for prefix, count in (("train", 108), ("val", 18)):
        (raw / "images" / prefix).mkdir(parents=True)
        rows = []
        for index in range(count):
            filename = f"{prefix}/{index + 1}.jpg"
            (raw / "images" / filename).write_bytes(f"{prefix}-opaque-image-{index}".encode())
            rows.append({"file": filename, "age": AGE_CLASSES[index // (12 if prefix == "train" else 2)],
                         "gender": "Female", "race": RACE_GROUPS[index % 7], "service_test": False})
        pd.DataFrame(rows).sample(frac=1, random_state=7).to_csv(
            raw / f"fairface_label_{prefix}.csv", index=False)
    for filename in ("train/1.jpg", "train/13.jpg", "val/2.jpg"):
        (raw / "images" / filename).write_bytes(b"internal-duplicate-also-in-holdout")
    for filename in ("train/25.jpg", "val/1.jpg"):
        (raw / "images" / filename).write_bytes(b"additional-holdout-overlap")
    for filename in ("val/3.jpg", "val/4.jpg"):
        (raw / "images" / filename).write_bytes(b"holdout-internal-duplicate-kept")
    return raw


class SplitPreparationChecks(unittest.TestCase):

    def test_original_exclusions_membership_serialization_and_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = publisher_fixture(root / "input")
            candidate = root / "candidate/splits/fairface-v1"
            metadata = prepare_splits(root / "input", candidate)
            self.assertEqual(metadata["counts"], {
                "publisher_train": 108, "excluded_s04_duplicates": 2, "excluded_duplicate_groups": 1,
                "train_images_matching_holdout_hashes": 3, "additional_overlap_exclusions": 1,
                "excluded_total": 3, "eligible": 105, "train": 94, "development": 11, "holdout": 18})
            self.assertEqual(metadata["image_hashing"]["labelled_images"], 126)
            self.assertFalse(metadata["image_hashing"]["decoded"])
            self.assertEqual(metadata["holdout_internal_duplicate_records"], 2)
            self.assertTrue(all(value == 0 for value in metadata["cross_partition_hash_overlaps"].values()))
            frames = {split: pd.read_csv(candidate / f"{split}.csv") for split in ("train", "development", "holdout")}
            # Known answer from the recovered original lexicographic-input,
            # age-stratified train_test_split call, seed 4531, fraction .1.
            self.assertEqual(frames["development"].sample_id.tolist(), [
                "train/100.jpg", "train/107.jpg", "train/18.jpg", "train/29.jpg", "train/38.jpg",
                "train/43.jpg", "train/60.jpg", "train/61.jpg", "train/74.jpg", "train/8.jpg", "train/89.jpg"])
            for frame in frames.values():
                self.assertEqual(frame.columns.tolist(), ["sample_id", "image_path", "age", "target", "race", "partition", "sha256"])
                self.assertEqual(frame.sample_id.tolist(), sorted(frame.sample_id))
                self.assertTrue(frame.image_path.eq("raw/fairface-padding-0.25/images/" + frame.sample_id).all())
            for first, second in (("train", "development"), ("train", "holdout"), ("development", "holdout")):
                self.assertFalse(set(frames[first].sample_id) & set(frames[second].sample_id))
                self.assertFalse(set(frames[first].sha256) & set(frames[second].sha256))
            excluded = pd.read_csv(candidate / "exclusions.csv").set_index("sample_id")
            self.assertEqual(set(excluded.index), {"train/1.jpg", "train/13.jpg", "train/25.jpg"})
            self.assertEqual(excluded.loc["train/1.jpg", "exclusion_reason"],
                             "S04_exact_byte_duplicate_group;exact_hash_overlap_with_publisher_validation")
            self.assertEqual(excluded.loc["train/25.jpg", "exclusion_reason"],
                             "exact_hash_overlap_with_publisher_validation")
            self.assertFalse(set(excluded.index) & set(pd.concat([frames["train"], frames["development"]]).sample_id))
            self.assertEqual(set(frames["holdout"].sample_id), set(pd.read_csv(raw / "fairface_label_val.csv").file))
            for name, details in metadata["files"].items():
                content = (candidate / name).read_bytes()
                self.assertNotIn(b"\r", content)
                self.assertEqual(hashlib.sha256(content).hexdigest(), details["sha256"])
            self.assertEqual(len(metadata["files"]), 8)
            self.assertEqual(json.loads((candidate / "metadata.json").read_text()), metadata)
            with patch.dict("fairage.data.SPLITS", {name: len(frame) for name, frame in frames.items()}):
                for split in frames:
                    pd.testing.assert_frame_equal(read_manifest(root / "candidate", split), frames[split])

    def test_existing_destination_refused_before_hashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing"
            output.mkdir()
            (output / "original.txt").write_text("preserve")
            with patch("fairage.data._sha256", side_effect=AssertionError("Must not hash")):
                with self.assertRaises(FileExistsError):
                    prepare_splits(root / "absent-input", output)
            self.assertEqual((output / "original.txt").read_text(), "preserve")

    def test_missing_image_creates_no_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = publisher_fixture(root / "input")
            (raw / "images/train/1.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_splits(root / "input", root / "candidate")
            self.assertFalse((root / "candidate").exists())

    def test_publisher_paths_cannot_escape_image_root(self):
        for filename in ("train/../../outside.jpg", r"train/..\..\outside.jpg",
                         r"train\..\..\outside.jpg"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw = publisher_fixture(root / "input")
                labels = pd.read_csv(raw / "fairface_label_train.csv")
                labels.loc[0, "file"] = filename
                labels.to_csv(raw / "fairface_label_train.csv", index=False)
                with patch("fairage.data._sha256", side_effect=AssertionError("Must not hash")):
                    with self.assertRaisesRegex(ValueError, "unique train/"):
                        prepare_splits(root / "input", root / "candidate")
                self.assertFalse((root / "candidate").exists())
