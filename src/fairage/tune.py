"""Optimize scratch and pretrained recipes through the shared FairAge trainer.

Persist sequential TPE studies in native Optuna journals, with development-F1
reporting, median pruning, a shared budget, and retained best checkpoints.

Main functions:
    sample_config: Suggest a recipe within the configured search space.
    run_trial: Train and report one candidate using development data.
    run_studies: Coordinate studies, allocation, and controller locking.
    study_status: Read native trial states and selected parameters.
"""

import math
from functools import partial
from pathlib import Path

import optuna
from filelock import FileLock
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from fairage.analysis import load_studies
from fairage.train import train


def sample_config(trial, base, model, search):
    """Select learning rates, regularization and modest augmentation, not architecture."""

    choices = search[model]
    config = dict(base, experiment=model)
    config["lr"] = trial.suggest_float("lr", *choices["lr"], log=True)
    config["weight_decay"] = trial.suggest_float("weight_decay", *choices["weight_decay"], log=True)
    for key in ("warmup_epochs", "schedule", "dropout", "label_smoothing", "augmentation"):
        config[key] = trial.suggest_categorical(key, choices[key])
    if model == "pretrained":
        config["head_lr_factor"] = trial.suggest_categorical("head_lr_factor", choices["head_lr_factor"])
    config.update(epochs=choices["epochs"], patience=choices["patience"],
                  min_epochs=choices["min_epochs"])
    return config


def remove_trial_checkpoints(directory):
    """Discard only bulky failed/pruned checkpoints; keep configuration and history."""

    for filename in ("best.pt", "latest.pt", "best.tmp", "latest.tmp"):
        (directory / filename).unlink(missing_ok=True)


def check_study_request(models, hours, allocation, prefix, trials):
    """Validate requested models/budget and normalize allocation weights."""

    if (not models or len(set(models)) != len(models)
            or any(m not in ("scratch", "pretrained") for m in models)):
        raise ValueError("Choose scratch and/or pretrained once each")
    allocation = allocation or [1.] * len(models)
    if (len(allocation) != len(models) or any(not math.isfinite(w) or w <= 0 for w in allocation)
            or not math.isfinite(hours) or hours <= 0 or (trials is not None and trials < 1)):
        raise ValueError("Use positive hours, allocation weights and trial cap")
    if not prefix or Path(prefix).name != prefix or prefix in (".", ".."):
        raise ValueError("Use a simple study prefix")
    return allocation


def prepare_study(storage, prefix, model, base, search, directory):
    """Load/create native settings, configure pruning, and fail abandoned trials."""

    study = optuna.create_study(
        study_name=f"{prefix}-{model}", storage=storage, direction="maximize",
        load_if_exists=True)
    if "config" not in study.user_attrs:
        study.set_user_attr("config", base)
        study.set_user_attr("search", search)
    saved, space = study.user_attrs["config"], study.user_attrs["search"]
    # Match the recorded Optuna 5.0.0 campaign defaults explicitly.
    # Restart retains journal history, not the sampler's advanced RNG state.
    study.sampler = optuna.samplers.TPESampler(
        seed=saved["seed"], n_startup_trials=space["startup_trials"],
        n_ei_candidates=24, multivariate=True, group=False, constant_liar=True)
    study.pruner = optuna.pruners.MedianPruner(
        n_startup_trials=space["pruner_startup_trials"],
        n_min_trials=space["pruner_min_trials"],
        n_warmup_steps=max(0, space[model]["grace_epochs"] - 1))
    # The caller holds the exclusive controller lock before inspecting RUNNING trials.
    for abandoned in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(abandoned.number, state=TrialState.FAIL)
        remove_trial_checkpoints(directory / "trials" / f"{study.study_name}-trial-{abandoned.number:04d}")
    return study, saved, space


def remaining_budget(study, seconds, trials):
    """Subtract native completed/pruned durations and count attempts toward the cap."""

    spent = sum(t.duration.total_seconds() for t in study.get_trials(
        states=(TrialState.COMPLETE, TrialState.PRUNED)) if t.duration is not None)
    remaining = max(0., seconds - spent)
    available_trials = None if trials is None else max(0, trials - len(study.trials))
    return remaining, available_trials


def run_trial(trial, base, model, search, directory):
    """Maximize a recipe's best development F1 using the shared trainer/precision.

    Optuna receives zero-based epoch steps; the callback can prune only after a
    completed epoch is saved. The trainer separately handles minimum epochs and
    patience. Pruned/failed checkpoints are removed; native records remain.
    No holdout information enters suggestions, pruning, or the objective.
    """

    config = sample_config(trial, base, model, search)
    run_id = f"{trial.study.study_name}-trial-{trial.number:04d}"

    def report(epoch, row):
        """Report development F1 and honor Optuna's pruning decision."""

        trial.report(row["dev_macro_f1"], step=epoch - 1)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Development pruning at epoch {epoch}")

    trial_directory = directory / "trials" / run_id
    if trial_directory.exists():
        raise ValueError(f"Trial output already exists: {trial_directory}")
    try:
        result = train(config, run_id, on_epoch=report, directory=trial_directory)
        if result["interrupted"]:
            raise KeyboardInterrupt("Owner requested an epoch-boundary stop")
        return result["best_score"]
    except BaseException:
        # Native records/config/history remain; pruned/failed trials are not resumed.
        remove_trial_checkpoints(trial_directory)
        raise


def optimize_study(prefix, model, base, search, directory, seconds, trials):
    """Prepare and optimize one study before the controller advances to the next."""

    directory.mkdir(exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(directory / "study.log"), JournalFileOpenLock(str(directory / "study.log"))))
    study, saved, space = prepare_study(storage, prefix, model, base, search, directory)
    remaining, available_trials = remaining_budget(study, seconds, trials)
    if remaining > 0 and available_trials != 0:
        study.optimize(
            partial(run_trial, base=saved, model=model, search=space, directory=directory),
            timeout=remaining, n_trials=available_trials, n_jobs=1,
            gc_after_trial=True)
    return study


def run_studies(base, search, models, hours, allocation=None, prefix="optuna", trials=None):
    """Run models sequentially under one controller lock and a shared target budget."""

    allocation = check_study_request(models, hours, allocation, prefix, trials)
    root = Path(base["run_root"]) / "tuning"
    root.mkdir(parents=True, exist_ok=True)
    results = []
    with FileLock(root / ".controller.lock", timeout=0):
        for model, weight in zip(models, allocation):
            directory = root / f"{prefix}-{model}"
            seconds = hours * 3600 * weight / sum(allocation)
            study = optimize_study(
                prefix, model, base, search, directory, seconds, trials)
            best = study.best_trial if study.get_trials(states=(TrialState.COMPLETE,)) else None
            results.append({"model": model, "study": study.study_name,
                            "journal": str(directory / "study.log"),
                            "trials": len(study.trials),
                            "best_checkpoint": str(directory / "trials" / f"{study.study_name}-trial-{best.number:04d}" / "best.pt")
                            if best else None})
    return results


def study_status(base, prefix="optuna"):
    """Display native Optuna states, best parameters and scores from the journal."""

    results = []
    for study in load_studies(base, prefix):
        states = {state.name: sum(t.state == state for t in study.trials)
                  for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.RUNNING)}
        best = study.best_trial if states["COMPLETE"] else None
        directory = Path(base["run_root"]) / "tuning" / study.study_name
        results.append({"study": study.study_name, "states": states,
                        "best_trial": best.number if best else None,
                        "best_config": str(directory / "trials" / f"{study.study_name}-trial-{best.number:04d}" / "config.json") if best else None,
                        "best_value": best.value if best else None,
                        "best_params": best.params if best else None})
    return results
