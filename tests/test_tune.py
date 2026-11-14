"""Tests Optuna study settings, scheduling, and trial results.

Purpose:
    Keep study settings, reported scores, selected checkpoints, and pruning behavior
    consistent.

Tests:
    test_original_horizons_scheduler_and_pruning_boundaries:
        Detects changes to the study's epoch limits, learning-rate schedules, or pruning
        boundaries.

    test_saved_sampler_settings_and_short_epoch_override:
        Detects resumed studies ignoring saved sampler settings or short runs retaining
        invalid warmup lengths.

    test_reporting_pruning_and_original_selected_artifacts:
        Detects incorrect trial reporting, wrong winning-trial paths, or retained
        checkpoints from pruned trials.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna
import torch
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from fairage.analysis import load_studies
from fairage.cli import main
from fairage.train import make_scheduler
from fairage.tune import prepare_study, run_studies, run_trial, sample_config, study_status


class TuningChecks(unittest.TestCase):
    def setUp(self):
        self.base = yaml.safe_load(Path("configs/baseline.yaml").read_text())
        self.search = yaml.safe_load(Path("configs/tuning.yaml").read_text())

    def test_original_horizons_scheduler_and_pruning_boundaries(self):
        for model, epochs, grace, minimum, patience in (("scratch", 100, 40, 40, 18), ("pretrained", 50, 10, 15, 12)):
            choices = self.search[model]
            self.assertEqual((choices["epochs"], choices["grace_epochs"], choices["min_epochs"], choices["patience"]),
                             (epochs, grace, minimum, patience))
            for warmup in choices["warmup_epochs"]:
                params = dict(lr=1e-4, weight_decay=.01, warmup_epochs=warmup, schedule="cosine",
                              dropout=0., label_smoothing=0., augmentation="standard", head_lr_factor=3.)
                config = sample_config(optuna.trial.FixedTrial(params), self.base, model, self.search)
                self.assertEqual((config["epochs"], config["min_epochs"], config["patience"]), (epochs, minimum, patience))
                optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=config["lr"])
                scheduler = make_scheduler(optimizer, config)
                for epoch in range(1, epochs + 1):
                    optimizer.step()
                    scheduler.step()
                    if warmup and epoch == warmup:
                        self.assertAlmostEqual(scheduler.get_last_lr()[0], config["lr"])
                    if not warmup and epoch == epochs // 2:
                        self.assertAlmostEqual(scheduler.get_last_lr()[0], config["lr"] / 2)
                self.assertEqual(scheduler.get_last_lr()[0], 0.)
            study, _, _ = prepare_study(optuna.storages.InMemoryStorage(), "boundary", model,
                                        self.base, self.search, Path("unused"))
            for _ in range(3):
                reference = study.ask()
                for step in (grace - 2, grace - 1):
                    reference.report(1., step=step)
                study.tell(reference, 1.)
            candidate = study.ask()
            candidate.report(0., step=grace - 2)
            self.assertFalse(candidate.should_prune())
            candidate.report(0., step=grace - 1)
            self.assertTrue(candidate.should_prune())

    def test_saved_sampler_settings_and_short_epoch_override(self):
        storage = optuna.storages.InMemoryStorage()
        prepare_study(storage, "saved", "scratch", self.base, self.search, Path("unused"))
        with patch("fairage.tune.optuna.samplers.TPESampler", wraps=optuna.samplers.TPESampler) as sampler:
            prepare_study(storage, "saved", "scratch", dict(self.base, seed=999),
                          dict(self.search, startup_trials=99), Path("unused"))
            self.assertEqual(sampler.call_args.kwargs, {
                "seed": 4531, "n_startup_trials": 5, "n_ei_candidates": 24,
                "multivariate": True, "group": False, "constant_liar": True})
        with patch("fairage.tune.run_studies", return_value=[]) as dispatch, contextlib.redirect_stdout(io.StringIO()):
            main(["tune", "--models", "scratch", "pretrained", "--epochs", "2"])
        base, search, models, *_ = dispatch.call_args.args
        for model in models:
            self.assertEqual(search[model]["epochs"], 2)
            self.assertEqual(search[model]["warmup_epochs"], [0])
        self.assertEqual(self.search["scratch"]["warmup_epochs"], [0, 5, 10])

    def test_reporting_pruning_and_original_selected_artifacts(self):
        attempts = []

        def train_candidate(config, run_id, on_epoch, directory):
            attempts.append(directory)
            directory.mkdir(parents=True)
            number = int(run_id.rsplit("-", 1)[1])
            (directory / "config.json").write_text(json.dumps(config))
            torch.save({"trial": number}, directory / "best.pt")
            score = .4 + .1 * number
            on_epoch(1, {"dev_macro_f1": score})
            on_epoch(2, {"dev_macro_f1": score - .1})
            return {"best_score": score, "interrupted": False}

        with tempfile.TemporaryDirectory() as temporary, patch("fairage.tune.train", side_effect=train_candidate):
            self.base["run_root"] = temporary
            result = run_studies(self.base, self.search, ["scratch"], hours=.01, prefix="integration", trials=2)[0]
            study = load_studies(self.base, "integration")[0]
            self.assertEqual(study.best_trial.number, 1)
            self.assertEqual(study.trials[0].intermediate_values, {0: .4, 1: .4 - .1})
            self.assertEqual(study.best_value, .5)
            checkpoint = Path(result["best_checkpoint"])
            self.assertEqual(torch.load(checkpoint, weights_only=True)["trial"], study.best_trial.number)
            self.assertEqual(Path(study_status(self.base, "integration")[0]["best_config"]), checkpoint.parent / "config.json")
            journal = str(Path(temporary) / "pruned.log")
            storage = JournalStorage(JournalFileBackend(journal, JournalFileOpenLock(journal)))
            pruned = optuna.create_study(study_name="pruned", storage=storage,
                                         direction="maximize", pruner=optuna.pruners.ThresholdPruner(lower=1.01))
            pruned.optimize(lambda trial: run_trial(trial, self.base, "scratch", self.search, Path(temporary)), n_trials=1)
            self.assertEqual(pruned.trials[0].state, TrialState.PRUNED)
            self.assertFalse((attempts[-1] / "best.pt").exists())
