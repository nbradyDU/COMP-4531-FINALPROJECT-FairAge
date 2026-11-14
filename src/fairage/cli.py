"""Expose FairAge training, tuning, evaluation, and saved-result commands.

All subcommands accept --config PATH (default: configs/baseline.yaml) and
-h/--help. Relative paths use the working directory. Unset overrides use
configuration values; resume and existing studies retain their saved settings.

Commands:
    train experiment --run RUN
        Start scratch, pretrained, or frozen training in a new named run.
        Accepts the shared overrides below and --lr RATE (baseline: 0.0001).
        Calls fairage.train.train (train.py).

    resume run
        Resume the named run from latest.pt and its saved configuration.
        Calls fairage.train.train with resume=True (train.py).

    status [run] [--tuning] [--study-prefix NAME]
        Inspect one run, or native Optuna journals with --tuning. A run is
        required unless --tuning is set. Study prefix defaults to optuna.
        Calls fairage.train.status (train.py) or fairage.tune.study_status
        (tune.py); the prefix applies only to tuning status.

    tune [model] [--models MODEL ...]
        Optimize scratch and/or pretrained recipes; default: both, in order.
        Use either the positional model or --models, not both.
        --hours HOURS: target campaign budget (72).
        --allocation WEIGHT ...: positive weights per selected model (equal).
        --trials N: total attempt cap per study (no cap by default).
        --study-prefix NAME: journal/run prefix (optuna).
        --search-config PATH: search spaces (configs/tuning.yaml).
        Also accepts the shared overrides below.
        Selected checkpoint/config paths point to the original winning trial.
        Calls fairage.tune.run_studies (tune.py).

    evaluate run [--checkpoint {best,latest}] [--split SPLIT] [--device DEVICE]
        Evaluate the named run's checkpoint in FP32 and export predictions.
        Checkpoint defaults to best; split is train/development/holdout and
        defaults to development. Device is cuda/cpu; unset uses the saved device.
        Calls fairage.evaluate.evaluate_checkpoint (evaluate.py).

    compare runs [runs ...] [--checkpoint {best,latest}] [--split SPLIT]
        Combine one or more saved evaluations; checkpoint defaults to best.
        Split is development/holdout (development). --name NAME selects a new
        directory under results_root (comparison); existing output is refused.
        Calls fairage.evaluate.compare_results (evaluate.py).

Shared train/tune overrides:
    --epochs N: maximum epochs; baseline default 50. New tuning trials use
        search-space caps (scratch 100/pretrained 50) unless explicitly set.
    --batch-size N: images per batch (64).
    --workers N: loader worker processes (8).
    --seed N: random seed (4531).
    --limit N: first N records per partition for verification (no limit).
    --device {cuda,cpu}: execution device (cuda); no automatic fallback.
    --precision {bf16,fp32}: training/monitoring precision (bf16).
    Parenthesized values are defaults from the supplied configurations;
    overriding --config or --search-config can change them.

Main functions:
    build_parser: Define public commands and configuration overrides.
    execute_command: Dispatch parsed arguments to the shared implementation.
    main: Load settings, handle errors, and print the command result.
"""

import argparse
import json
from pathlib import Path

import yaml

def add_training_arguments(parser, include_lr=False):
    """Define the shared numeric/device overrides used by training and tuning."""

    options = [("epochs", int), ("batch-size", int), ("workers", int),
               ("seed", int), ("limit", int)]
    if include_lr:
        options.insert(3, ("lr", float))
    for name, kind in options:
        parser.add_argument("--" + name, type=kind)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--precision", choices=("bf16", "fp32"))


def add_tuning_parser(commands):
    """Define the model selection, budget, storage prefix and search-file arguments."""

    parser = commands.add_parser("tune", help="run sequential native Optuna studies")
    parser.add_argument("model", choices=("scratch", "pretrained"), nargs="?")
    parser.add_argument("--models", choices=("scratch", "pretrained"), nargs="+")
    parser.add_argument("--hours", type=float, default=72.)
    parser.add_argument("--allocation", type=float, nargs="+")
    parser.add_argument("--study-prefix", default="optuna")
    parser.add_argument("--trials", type=int, help="optional total trial cap per study")
    parser.add_argument("--search-config", type=Path, default=Path("configs/tuning.yaml"))
    add_training_arguments(parser)
    return parser


def build_parser():
    """Describe the existing public commands without performing study operations."""

    parser = argparse.ArgumentParser(prog="fairage")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train one baseline condition")
    training.add_argument("experiment", choices=("scratch", "pretrained", "frozen"))
    training.add_argument("--run", required=True)
    add_training_arguments(training, include_lr=True)
    monitoring = commands.add_parser("status")
    monitoring.add_argument("run", nargs="?")
    monitoring.add_argument("--tuning", action="store_true")
    monitoring.add_argument("--study-prefix", default="optuna")
    commands.add_parser("resume").add_argument("run")
    evaluation = commands.add_parser("evaluate", help="save checkpoint predictions and metrics")
    evaluation.add_argument("run")
    evaluation.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    evaluation.add_argument("--split", choices=("train", "development", "holdout"), default="development")
    evaluation.add_argument("--device", choices=("cuda", "cpu"))
    comparison = commands.add_parser("compare", help="compare saved results")
    comparison.add_argument("runs", nargs="+")
    comparison.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    comparison.add_argument("--split", choices=("development", "holdout"), default="development")
    comparison.add_argument("--name", default="comparison")
    tuning = add_tuning_parser(commands)
    for subparser in (training, monitoring, commands.choices["resume"], evaluation,
                      comparison, tuning):
        subparser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    return parser


def execute_command(args, config):
    """Apply command-specific arguments and call the corresponding ordinary function."""

    if args.command == "train":
        from fairage.train import train

        config.update({k: v for k, v in vars(args).items()
                       if v is not None and k not in ("command", "run", "config")})
        return train(config, args.run)
    if args.command == "resume":
        from fairage.train import train

        return train(config, args.run, resume=True)
    if args.command == "status":
        if args.tuning:
            from fairage.tune import study_status

            return study_status(config, args.study_prefix)
        if args.run is None:
            raise ValueError("Supply a run name or --tuning")
        from fairage.train import status

        return status(config, args.run)
    if args.command == "evaluate":
        from fairage.evaluate import evaluate_checkpoint

        return evaluate_checkpoint(config, args.run, args.checkpoint, args.split, args.device)
    if args.command == "tune":
        from fairage.tune import run_studies

        if args.model and args.models:
            raise ValueError("Use a positional model or --models, not both")
        models = args.models or ([args.model] if args.model else ["scratch", "pretrained"])
        for key in ("epochs", "batch_size", "workers", "seed", "limit", "device", "precision"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        search = yaml.safe_load(args.search_config.read_text())
        if args.epochs is not None:
            for model in models:
                search[model]["epochs"] = args.epochs
                search[model]["warmup_epochs"] = [
                    warmup for warmup in search[model]["warmup_epochs"] if warmup < args.epochs]
                if not search[model]["warmup_epochs"]:
                    raise ValueError("Epoch override must allow a warmup shorter than training")
        return run_studies(config, search, models, args.hours, args.allocation,
                           args.study_prefix, args.trials)
    from fairage.evaluate import compare_results

    return compare_results(config, args.runs, args.checkpoint, args.split, args.name)


def main(argv=None):
    """Load arguments/settings, dispatch, and report the result or a clear error."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(args.config.read_text())
        result = execute_command(args, config)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0
