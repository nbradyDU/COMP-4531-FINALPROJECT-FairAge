"""Prepare frozen FairFace partitions and load model inputs.

Verify membership and age labels, apply RGB preprocessing, and seed training
augmentation without modifying source artifacts.

Main entry points:
    prepare_splits: Hash publisher images and explicitly prepare frozen partitions.
    read_manifest: Verify frozen membership and nine-class target mapping.
    FairFaceDataset: Read images and apply partition-specific transforms.
    make_loaders: Build seeded training and development loaders.
"""

import argparse
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49",
               "50-59", "60-69", "more than 70"]
RACE_GROUPS = ["East Asian", "Indian", "Black", "White", "Middle Eastern",
               "Latino_Hispanic", "Southeast Asian"]
SPLITS = {"train": 77805, "development": 8645, "holdout": 10954}


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest(frame, partition):
    """Preserve the original identifiers, column order and lexicographic output."""

    result = frame.rename(columns={"file": "sample_id"}).copy()
    result["image_path"] = "raw/fairface-padding-0.25/images/" + result["sample_id"]
    result["target"] = result["age"].map(dict(zip(AGE_CLASSES, range(9))))
    if result.target.isna().any() or not result.race.isin(RACE_GROUPS).all():
        raise ValueError(f"{partition}: unknown age or annotated-race label")
    result["target"] = result.target.astype(int)
    result["partition"] = partition
    columns = ["sample_id", "image_path", "age", "target", "race", "partition", "sha256"]
    if "exclusion_reason" in result:
        columns.append("exclusion_reason")
    return result[columns].sort_values("sample_id").reset_index(drop=True)


def prepare_splits(data_root, output_dir=None):
    """Prepare the original seed-4531 age-stratified split from raw image bytes.

    Read publisher labels and images under raw/fairface-padding-0.25/. Hash each
    labelled image once without decoding it. Exclude every internal training
    duplicate and remaining holdout-hash match. Image paths in all eight CSVs
    remain relative to data_root. Return metadata for the existing loader.

    Default output is data_root/splits/fairface-v1. An explicit output_dir permits
    isolated verification. Refuse any existing destination before hashing; never
    overwrite a freeze or write inside the raw source directory.
    """

    data_root = Path(data_root)
    raw = data_root / "raw/fairface-padding-0.25"
    output = Path(output_dir) if output_dir is not None else data_root / "splits/fairface-v1"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Split destination already exists; choose a new directory: {output}")
    if output.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Split output must be outside the raw source directory")
    train_path, holdout_path = raw / "fairface_label_train.csv", raw / "fairface_label_val.csv"

    hashes = []
    for prefix, path in (("train", train_path), ("val", holdout_path)):
        files = pd.read_csv(path, usecols=["file"])["file"]
        if (files.isna().any() or files.duplicated().any()
                or not files.str.fullmatch(rf"{prefix}/[^/\\]+\.jpg").all()):
            raise ValueError(f"Expected unique {prefix}/ JPEG filenames in {path.name}")
        hashes.extend((name, _sha256(raw / "images" / name)) for name in files)
    inventory = pd.DataFrame(hashes, columns=["file", "sha256"])
    training_hashes = inventory[inventory.file.str.startswith("train/")]
    holdout_hashes = inventory[inventory.file.str.startswith("val/")]
    duplicates = training_hashes[training_hashes.duplicated("sha256", keep=False)]

    # Membership/serialization restore the producer at 060f663:src/fairage/data.py,
    # SHA-256 f58bbd5aaab327950ca74e74df4d069df806c0df06bae865c3066f989c3414b1.
    # Fresh byte hashes replace its private S03/S04 inventory dependencies.
    source = pd.read_csv(train_path).merge(training_hashes, on="file", validate="one_to_one")
    source = source.sort_values("file").reset_index(drop=True)
    duplicate_mask = source.file.isin(duplicates.file)
    overlap_mask = source.sha256.isin(holdout_hashes.sha256)
    excluded = source[duplicate_mask | overlap_mask].copy()
    excluded["exclusion_reason"] = ""
    excluded.loc[excluded.file.isin(duplicates.file), "exclusion_reason"] = "S04_exact_byte_duplicate_group"
    excluded.loc[excluded.sha256.isin(holdout_hashes.sha256), "exclusion_reason"] += ";exact_hash_overlap_with_publisher_validation"
    excluded["exclusion_reason"] = excluded.exclusion_reason.str.strip(";")
    eligible = source[~(duplicate_mask | overlap_mask)]
    training, development = train_test_split(
        eligible, test_size=0.10, random_state=4531, stratify=eligible.age)
    # Holdout labels do not determine training membership or stratification.
    holdout = pd.read_csv(holdout_path).merge(holdout_hashes, on="file", validate="one_to_one")
    frames = {"train": training, "development": development, "holdout": holdout}
    overlaps = {}
    for first, second in (("train", "development"), ("train", "holdout"), ("development", "holdout")):
        overlap = len(set(frames[first].sha256) & set(frames[second].sha256))
        if overlap:
            raise ValueError(f"Exact image-hash overlap remains: {first}/{second}")
        overlaps[f"{first}_{second}"] = overlap
    tables = {f"{name}.csv": _manifest(frame, name) for name, frame in frames.items()}
    tables["exclusions.csv"] = _manifest(excluded, "excluded")
    support_frames = {
        "publisher_train": source, "excluded_duplicates": source[duplicate_mask],
        "excluded_holdout_overlap_additional": source[overlap_mask & ~duplicate_mask],
        "excluded_total": excluded, "eligible": eligible, "train": training, "development": development,
    }
    for field, order in (("age", AGE_CLASSES), ("race", RACE_GROUPS)):
        support = pd.DataFrame({name: frame[field].value_counts().reindex(order, fill_value=0)
                                for name, frame in support_frames.items()})
        support.index.name = field
        tables[f"{field}_support.csv"] = support.reset_index()
    for name, frame in (("train", training), ("development", development)):
        tables[f"{name}_age_race_counts.csv"] = pd.crosstab(frame.age, frame.race).reindex(
            index=AGE_CLASSES, columns=RACE_GROUPS, fill_value=0).reset_index()

    contents = {name: frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
                for name, frame in tables.items()}
    metadata = {
        "schema_version": 1, "dataset": "FairFace aligned padding=0.25",
        "image_path_base": ".", "age_classes": AGE_CLASSES,
        "algorithm": {"name": "sklearn.model_selection.train_test_split / StratifiedShuffleSplit",
                      "seed": 4531, "development_fraction": 0.10, "stratify": "original age label",
                      "input_order": "lexicographic publisher file", "output_order": "lexicographic sample_id",
                      "versions": {name: version(name) for name in ("scikit-learn", "numpy", "pandas")}},
        "source_sha256": {path.name: _sha256(path) for path in (train_path, holdout_path)},
        "implementation_sha256": _sha256(Path(__file__)),
        "image_hashing": {"algorithm": "sha256", "labelled_images": len(inventory), "decoded": False},
        "counts": {"publisher_train": len(source), "excluded_s04_duplicates": len(duplicates),
                   "excluded_duplicate_groups": int(duplicates.sha256.nunique()),
                   "train_images_matching_holdout_hashes": int(overlap_mask.sum()),
                   "additional_overlap_exclusions": int((overlap_mask & ~duplicate_mask).sum()),
                   "excluded_total": len(excluded), "eligible": len(eligible),
                   **{name: len(frame) for name, frame in frames.items()}},
        "cross_partition_hash_overlaps": overlaps,
        "holdout_internal_duplicate_records": int(holdout_hashes.duplicated("sha256", keep=False).sum()),
        "primary_training_policy": {"shuffle": True, "loss": "unweighted_cross_entropy",
                                    "automatic_age_or_race_balancing": False,
                                    "shared_initializations": ["scratch", "pretrained"]},
        "files": {name: {"rows": len(tables[name]), "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in contents.items()},
        "limitations": ["Exact byte hashes do not establish near-duplicate or person-disjointness.",
                        "Publisher holdout retains its internal duplicates."],
    }
    contents["metadata.json"] = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    return metadata


def read_manifest(data_root, split):
    """Read one frozen partition and reject changed membership or class mapping."""

    if split not in SPLITS:
        raise ValueError("Choose train, development, or holdout")
    directory = Path(data_root) / "splits/fairface-v1"
    metadata = json.loads((directory / "metadata.json").read_text())
    path = directory / f"{split}.csv"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if metadata["age_classes"] != AGE_CLASSES or digest != metadata["files"][path.name]["sha256"]:
        raise ValueError("Frozen classes or manifest changed")
    rows = pd.read_csv(path)
    expected = rows.age.map(dict(zip(AGE_CLASSES, range(9))))
    if len(rows) != SPLITS[split] or not expected.equals(rows.target):
        raise ValueError("Frozen split size or target mapping is invalid")
    return rows


class FairFaceDataset(Dataset):
    """Load RGB images; augmentation is applied only to training images in RAM."""

    def __init__(self, data_root, split, augmentation="standard", limit=None, *, augment=True):
        """Read one partition; augment=False gives deterministic preprocessing on any split."""

        self.root = Path(data_root)
        self.rows = read_manifest(data_root, split)
        if limit is not None:
            self.rows = self.rows.head(limit)
        training = split == "train" and augment
        operations = [transforms.Resize(248, interpolation=InterpolationMode.BICUBIC,
                                       antialias=True)]
        if training and augmentation == "crop":
            operations.append(transforms.RandomResizedCrop(
                224, scale=(.9, 1.0), ratio=(.95, 1.05),
                interpolation=InterpolationMode.BICUBIC, antialias=True))
        else:
            operations.append(transforms.CenterCrop(224))
        if training:
            operations.append(transforms.RandomHorizontalFlip(.5))
        operations += [transforms.ToTensor(),
                       transforms.Normalize((.485, .456, .406), (.229, .224, .225))]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        """Return the selected manifest size."""

        return len(self.rows)

    def __getitem__(self, index):
        """Return a normalized RGB tensor and its integer age-group target."""

        row = self.rows.iloc[index]
        with Image.open(self.root / row.image_path) as image:
            return self.transform(image.convert("RGB")), int(row.target)


def seed_worker(worker_id):
    """Seed Python/NumPy augmentation consumers from PyTorch's worker seed."""

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(config):
    """Build separate train/development loaders; never load the held-out split."""

    loaders = []
    for split, offset in (("train", 0), ("development", 1)):
        dataset = FairFaceDataset(config["data_root"], split, config["augmentation"],
                                  config.get("limit"))
        loaders.append(DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=split == "train",
            num_workers=config["workers"], worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config["seed"] + offset),
            pin_memory=config["device"] == "cuda"))
    return tuple(loaders)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explicitly prepare the original FairFace age-stratified splits.")
    parser.add_argument("command", choices=["prepare-splits"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, help="New destination; defaults to DATA_ROOT/splits/fairface-v1")
    arguments = parser.parse_args()
    prepared = prepare_splits(arguments.data_root, arguments.output_dir)
    print(json.dumps(prepared["counts"], indent=2))
