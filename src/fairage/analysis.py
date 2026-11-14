"""Read saved run histories, comparable evaluation tables, and Optuna journals."""

import json
from pathlib import Path

import pandas as pd


def find_run(config, run_id):
    """Locate one unique baseline or trial ID under the configured run root."""

    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Use a simple run name")
    root = Path(config["run_root"])
    candidates = [*root.glob(f"baselines/*/{run_id}"),
                  *root.glob(f"tuning/*/trials/{run_id}")]
    candidates = [path for path in candidates if path.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one run named {run_id}; found {len(candidates)}")
    return candidates[0]


def load_results(config, run_ids, checkpoint="best", split="development"):
    """Combine metrics, histories and subgroup tables for matched populations.

    Predictions supply sample/target/race identity only; probability columns and
    confusion tables are not needed for comparisons. No checkpoint is loaded.
    """

    if checkpoint not in ("best", "latest") or split not in ("train", "development", "holdout"):
        raise ValueError("Unknown checkpoint or partition")
    tables = {key: [] for key in ("metrics", "history", "by_age", "by_race", "by_age_race")}
    reference, precision = None, None
    for run_id in run_ids:
        directory = find_run(config, run_id)
        saved = json.loads((directory / "config.json").read_text())
        history = pd.read_csv(directory / "history.csv")
        selected = history.loc[history.dev_macro_f1.idxmax()] if checkpoint == "best" else history.iloc[-1]
        epoch = int(selected.epoch)
        output = directory / "evaluation" / f"{checkpoint}-{split}-e{epoch}"
        summary = json.loads((output / "metrics.json").read_text())
        if (summary["run"], summary["checkpoint"], summary["split"], summary["epoch"]) != (run_id, checkpoint, split, epoch):
            raise ValueError("Saved evaluation identity does not match the selected checkpoint/partition")
        samples = pd.read_csv(output / "predictions.csv", usecols=["sample_id", "target", "race"])
        samples = samples.sort_values("sample_id").reset_index(drop=True)
        if samples.sample_id.duplicated().any() or len(samples) != summary["count"]:
            raise ValueError("Evaluation sample records are duplicated or incomplete")
        if reference is not None and (not reference.equals(samples) or precision != summary["precision"]):
            raise ValueError("Comparisons require identical evaluated samples/labels and precision")
        reference, precision = samples, summary["precision"]
        tables["metrics"].append(pd.DataFrame([{k: v for k, v in summary.items() if k != "config"}]))
        tables["history"].append(history.assign(run=run_id, experiment=saved["experiment"]))
        for name in ("by_age", "by_race", "by_age_race"):
            tables[name].append(pd.read_csv(output / f"{name}.csv").assign(run=run_id))
    if not tables["metrics"]:
        raise ValueError("Select at least one evaluated run")
    return {key: pd.concat(values, ignore_index=True) for key, values in tables.items()}


def load_studies(config, prefix="optuna"):
    """Read native scratch/pretrained journals for the selected study prefix."""

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    studies = []
    for model in ("scratch", "pretrained"):
        name = f"{prefix}-{model}"
        journal = Path(config["run_root"]) / "tuning" / name / "study.log"
        if journal.is_file():
            storage = JournalStorage(JournalFileBackend(str(journal), JournalFileOpenLock(str(journal))))
            studies.append(optuna.load_study(study_name=name, storage=storage))
    if not studies:
        raise ValueError("No journals exist for this study prefix")
    return studies
