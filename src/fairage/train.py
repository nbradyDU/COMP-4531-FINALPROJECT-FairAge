"""Train and manage FairAge DeiT-Tiny experiments.

Share optimization, epoch metrics, checkpoint selection, and interrupted-run
recovery across scratch, pretrained, and frozen conditions.

Main functions:
    configure_model: Select initialization and trainable parameters.
    train: Start or resume a run using the shared epoch loop.
    save_epoch: Save completed metrics, checkpoints, and continuation state.
    status: Report saved progress and the run's lock state.
"""

import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock, Timeout

from fairage.analysis import find_run
from fairage.data import make_loaders
from fairage.evaluate import evaluate_loader
from fairage.metrics import classification_metrics
from fairage.model import DeiTTiny, load_pretrained_weights


def check_config(config):
    """Reject invalid numerical settings and unsupported training choices."""

    for key in ("epochs", "batch_size", "lr", "gradient_clip"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("weight_decay", "head_lr_factor", "dropout", "label_smoothing"):
        if not np.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if config["workers"] < 0 or config["weight_decay"] < 0 or config["head_lr_factor"] <= 0:
        raise ValueError("Invalid workers, weight decay or head learning-rate factor")
    if not 0 <= config["dropout"] < 1 or not 0 <= config["label_smoothing"] < 1:
        raise ValueError("Dropout and label smoothing must be in [0,1)")
    if not 0 <= config["warmup_epochs"] < config["epochs"]:
        raise ValueError("Warmup must be shorter than training")
    if config["schedule"] not in ("cosine", "constant") or config["augmentation"] not in ("standard", "crop"):
        raise ValueError("Unknown schedule or augmentation")
    if config["precision"] not in ("bf16", "fp32") or config["device"] not in ("cpu", "cuda"):
        raise ValueError("Choose fp32/bf16 and cpu/cuda")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise ValueError("Use fp32 for CPU verification")
    if config.get("limit") is not None and config["limit"] < 1:
        raise ValueError("A smoke-test limit must be positive")
    if config["patience"] < 0 or config["min_epochs"] < 0:
        raise ValueError("Invalid early stopping settings")


def train_one_epoch(model, loader, optimizer, config):
    """Update trainable parameters and return loss, accuracy, F1 and epoch seconds."""

    model.train()
    actual, predicted = [], []
    total_loss = 0.0
    if config["device"] == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for images, targets in loader:
        images, targets = images.to(config["device"]), targets.to(config["device"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(config["device"], dtype=torch.bfloat16,
                            enabled=config["precision"] == "bf16"):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(
                logits, targets, label_smoothing=config["label_smoothing"])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"],
                                       error_if_nonfinite=True)
        optimizer.step()
        total_loss += loss.item() * len(targets)
        actual.extend(targets.tolist())
        predicted.extend(logits.detach().argmax(1).tolist())
    if config["device"] == "cuda":
        torch.cuda.synchronize()
    return {"loss": total_loss / len(actual), **classification_metrics(actual, predicted),
            "seconds": time.perf_counter() - started}


def make_optimizer(model, config):
    """Use AdamW on trainable weights; optionally raise the pretrained head LR.

    Frozen mode exposes only head parameters. Full fine-tuning can place the
    backbone and head in separate groups; weight decay applies to both.
    """

    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = trainable
    if config["experiment"] == "pretrained" and config["head_lr_factor"] != 1:
        groups = [
            {"params": [p for n, p in model.named_parameters() if not n.startswith("head.")],
             "lr": config["lr"]},
            {"params": model.head.parameters(), "lr": config["lr"] * config["head_lr_factor"]}]
    return torch.optim.AdamW(groups, lr=config["lr"], weight_decay=config["weight_decay"])


def make_scheduler(optimizer, config):
    """Apply optional linear warmup, followed by cosine decay or a constant rate."""

    def multiplier(epoch):
        """Return the warmup/decay factor for an epoch index."""

        warmup = config["warmup_epochs"]
        if epoch < warmup:
            return (epoch + 1) / warmup
        if config["schedule"] == "constant":
            return 1.0
        return .5 * (1 + math.cos(math.pi * (epoch - warmup) /
                                 (config["epochs"] - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_checkpoint(path, state):
    """Replace a single completed checkpoint without exposing a partial torch.save."""

    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def random_state(loaders):
    """Capture randomness needed for continuation at an epoch boundary."""

    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loaders": [loader.generator.get_state() for loader in loaders]}


def restore_random_state(state, loaders):
    """Restore saved seeds/order after model and optimizer construction."""

    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    for loader, generator in zip(loaders, state["loaders"]):
        loader.generator.set_state(generator)


def load_training_state(config, directory, resume):
    """Read saved configuration for resume, or keep the new run's settings."""

    if not resume:
        return config, None
    state = torch.load(directory / "latest.pt", map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint; historical formats are not migrated")
    config = state["config"]
    check_config(config)
    return config, state


def configure_model(config, initialize=True):
    """Select scratch/full/frozen training; resume skips ImageNet reinitialization."""

    experiment = config["experiment"]
    if experiment not in ("scratch", "pretrained", "frozen"):
        raise ValueError("Unknown experiment")
    model = DeiTTiny(config["dropout"])
    if initialize and experiment in ("pretrained", "frozen"):
        load_pretrained_weights(model, config["weights"])
    if experiment == "frozen":
        model.requires_grad_(False)
        model.head.requires_grad_(True)
    return model


def prepare_model_and_data(config, initialize):
    """Seed the run, configure its model/trainability, and create train/dev loaders."""

    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no automatic CPU fallback")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = configure_model(config, initialize)
    model.to(config["device"])
    return model, make_loaders(config)


def restore_training_state(model, optimizer, scheduler, loaders, state):
    """Restore weights, optimizer, schedule and randomness after trainability is set."""

    if state is None:
        return [], -1.0, 0
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_random_state(state["rng"], loaders)
    return state["history"], state["best_score"], state["best_epoch"]


def save_epoch(directory, config, model, optimizer, scheduler, loaders,
               history, best_score, best_epoch):
    """Save a completed epoch with next-epoch schedule and continuation RNG state.

    Strictly greater development macro-F1 replaces best, preserving the earliest
    exact tie. Training precision governs selection; later FP32 evaluation does
    not reselect. Each checkpoint replacement is atomic, not the whole file set.
    """

    row = history[-1]
    improved = row["dev_macro_f1"] > best_score
    if improved:
        best_score, best_epoch = row["dev_macro_f1"], row["epoch"]
    scheduler.step()
    state = {"format_version": 1, "config": config, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": row["epoch"], "history": history, "best_score": best_score,
             "best_epoch": best_epoch, "rng": random_state(loaders)}
    # Publish a new best before latest can advance past it.
    if improved:
        save_checkpoint(directory / "best.pt", state)
    save_checkpoint(directory / "latest.pt", state)
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    return best_score, best_epoch


def run_training_epochs(model, loaders, optimizer, scheduler, config, directory,
                        history, best_score, best_epoch, on_epoch):
    """Run epochs with the existing callback, early-stop and owner-signal behavior."""

    stop = False
    interrupted = False

    def request_stop(signum, frame):
        """Defer owner signals until the current epoch checkpoint is complete."""

        nonlocal stop, interrupted
        stop = interrupted = True

    handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for epoch in range(len(history) + 1, config["epochs"] + 1):
            lr = optimizer.param_groups[0]["lr"]
            training = train_one_epoch(model, loaders[0], optimizer, config)
            development, _ = evaluate_loader(
                model, loaders[1], config["device"], config["precision"])
            history.append({"epoch": epoch, "lr": lr,
                            "head_lr": optimizer.param_groups[-1]["lr"],
                            **{f"train_{k}": v for k, v in training.items()},
                            **{f"dev_{k}": v for k, v in development.items()}})
            best_score, best_epoch = save_epoch(
                directory, config, model, optimizer, scheduler, loaders,
                history, best_score, best_epoch)
            print(json.dumps(history[-1]), flush=True)
            if interrupted:
                break
            if on_epoch is not None and on_epoch(epoch, history[-1]) is False:
                stop = True
            if (config["patience"] and epoch >= config["min_epochs"]
                    and epoch - best_epoch >= config["patience"]):
                stop = True
            if stop:
                break
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return {"epoch": len(history), "best_score": best_score, "best_epoch": best_epoch,
            "stopped": stop, "interrupted": interrupted}


def train(config, run_id, resume=False, on_epoch=None, directory=None):
    """Start or resume one run under an exclusive process lock.

    New runs create a unique output directory from config. Resume restores
    latest.pt and its saved configuration, optimizer, schedule, and RNG state.
    on_epoch receives each saved epoch and may stop training by returning False.
    Return completed progress; signals request a stop at an epoch boundary.
    """

    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Use a simple run name")
    if directory is None:
        directory = (find_run(config, run_id) if resume else
                     Path(config["run_root"]) / "baselines" / config["experiment"] / run_id)
    directory = Path(directory)
    if not resume:
        check_config(config)
        root = Path(config["run_root"])
        if list(root.glob(f"baselines/*/{run_id}")) or list(root.glob(f"tuning/*/trials/{run_id}")):
            raise ValueError(f"Run ID already exists: {run_id}")
        directory.mkdir(parents=True, exist_ok=False)
    with FileLock(directory / ".lock", timeout=0):
        config, state = load_training_state(config, directory, resume)
        model, loaders = prepare_model_and_data(config, initialize=not resume)
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        history, best_score, best_epoch = restore_training_state(
            model, optimizer, scheduler, loaders, state)
        if state is None:
            config = dict(config, trainable_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad))
            (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        result = run_training_epochs(
            model, loaders, optimizer, scheduler, config, directory,
            history, best_score, best_epoch, on_epoch)
        return {"run": run_id, **result}


def status(config, run_id):
    """Read last saved progress and whether another process holds the run lock."""

    directory = find_run(config, run_id)
    try:
        with FileLock(directory / ".lock", timeout=0):
            active = False
    except Timeout:
        active = True
    if not (directory / "latest.pt").exists():
        return {"run": run_id, "active": active, "epoch": 0,
                "note": "No completed epoch checkpoint yet"}
    state = torch.load(directory / "latest.pt", map_location="cpu", weights_only=True)
    return {"run": run_id, "active": active, "epoch": state["epoch"],
            "planned_epochs": state["config"]["epochs"], "best_epoch": state["best_epoch"],
            "best_dev_macro_f1": state["best_score"], "last_epoch": state["history"][-1]}
