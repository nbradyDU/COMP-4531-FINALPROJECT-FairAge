"""Evaluate age classifiers and export scientific metrics, tables, and figures.

Use the configured precision for training-time monitoring and deterministic
FP32 for explicit checkpoint evaluation. Derived comparisons reuse saved exports.
Metric definitions live in fairage.metrics.

Main functions:
    evaluate_loader: Return loss, classification scores, and probabilities.
    save_analysis: Export confusion, per-age, and demographic tables.
    evaluate_checkpoint: Save one checkpoint/partition evaluation.
    compare_results: Combine compatible saved results and draw comparisons.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from fairage.analysis import find_run, load_results
from fairage.data import AGE_CLASSES, RACE_GROUPS, FairFaceDataset
from fairage.metrics import classification_metrics, confidence_metrics, scientific_metrics
from fairage.model import DeiTTiny


def evaluate_loader(model, loader, device, precision="fp32"):
    """Return metrics and [N,9] probabilities in the loader's iteration order.

    Loss is sample-weighted, unsmoothed cross-entropy on logits. Training calls
    this with its configured precision; explicit checkpoint evaluation uses the
    FP32 default. Logits become float32 before softmax, including the BF16 path.
    The caller supplies a nonempty loader and controls preprocessing/order.
    """

    model.eval()
    actual, probabilities = [], []
    total_loss = 0.0
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(images.to(device))
                loss = torch.nn.functional.cross_entropy(logits, targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite development loss")
            total_loss += loss.item() * len(targets)
            probabilities.append(logits.float().softmax(1).cpu().numpy())
            actual.extend(targets.tolist())
    if device == "cuda":
        torch.cuda.synchronize()
    values = np.concatenate(probabilities)
    metrics = {"loss": total_loss / len(actual),
               **classification_metrics(actual, values.argmax(1)),
               "seconds": time.perf_counter() - started}
    return metrics, values


def save_analysis(records, directory):
    """Write confusion, per-age and annotated subgroup tables from saved predictions."""

    actual, predicted = records.target, records.predicted
    matrix = confusion_matrix(actual, predicted, labels=range(9))
    pd.DataFrame(matrix, index=AGE_CLASSES, columns=AGE_CLASSES).to_csv(directory / "confusion.csv")
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
    pd.DataFrame(normalized, index=AGE_CLASSES, columns=AGE_CLASSES).to_csv(directory / "confusion_normalized.csv")
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=range(9), zero_division=0)
    pd.DataFrame({"age": AGE_CLASSES, "support": support, "precision": precision,
                  "recall": recall, "f1": f1}).to_csv(directory / "by_age.csv", index=False)
    races = RACE_GROUPS
    groups, intersections = [], []
    for race in races:
        rows = records[records.race == race]
        groups.append({"race": race, "support": len(rows),
                       **scientific_metrics(rows.target, rows.predicted)})
        for age, label in enumerate(AGE_CLASSES):
            cell = rows[rows.target == age]
            intersections.append({"race": race, "age": label, "support": len(cell),
                                  "accuracy": float((cell.target == cell.predicted).mean())
                                  if len(cell) else None})
    pd.DataFrame(groups).to_csv(directory / "by_race.csv", index=False)
    pd.DataFrame(intersections).to_csv(directory / "by_age_race.csv", index=False)


def save_evaluation_figures(directory, history):
    """Draw the standard evaluation figures from saved tables and recorded epochs."""

    import matplotlib.pyplot as plt
    from fairage import plots

    matrix = pd.read_csv(directory / "confusion.csv", index_col=0)
    age = pd.read_csv(directory / "by_age.csv")
    race = pd.read_csv(directory / "by_race.csv").assign(run=history.run.iloc[0])
    cells = pd.read_csv(directory / "by_age_race.csv")
    for figure, _ in (
        plots.plot_confusion_matrix(matrix, save=True, save_path=directory / "confusion.png"),
        plots.plot_confusion_matrix(matrix, normalized=True, save=True,
                                    save_path=directory / "confusion_normalized.png"),
        plots.plot_age_metrics(age, save=True, save_path=directory / "per_age_metrics.png"),
        plots.plot_demographic_accuracy(race, save=True, save_path=directory / "demographic_accuracy.png"),
        plots.plot_age_race_support(cells, save=True, save_path=directory / "age_race_support.png"),
        plots.plot_learning_curves(history, save=True, save_path=directory / "learning_curves.png"),
        plots.plot_loss_curves(history, save=True, save_path=directory / "loss_curves.png"),
        plots.plot_training_efficiency(history, save=True, save_path=directory / "epoch_timing.png"),
    ):
        plt.close(figure)


def evaluate_checkpoint(config, run_id, checkpoint="best", split="development", device=None):
    """Evaluate a saved best/latest checkpoint deterministically in FP32.

    Use saved preprocessing settings and device unless device is explicit.
    Write predictions, metrics, subgroup tables, and figures into a new
    checkpoint/split/epoch directory; an existing output is not overwritten.
    Return the output path and aggregate scientific metrics.
    """

    if checkpoint not in ("best", "latest") or split not in ("train", "development", "holdout"):
        raise ValueError("Unknown checkpoint or partition")
    path = find_run(config, run_id) / f"{checkpoint}.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    saved = state["config"]
    device = device or saved["device"]
    dataset = FairFaceDataset(saved["data_root"], split, limit=saved.get("limit"), augment=False)
    loader = DataLoader(dataset, batch_size=saved["batch_size"], num_workers=saved["workers"])
    model = DeiTTiny(saved["dropout"])
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    metrics, probabilities = evaluate_loader(model, loader, device)
    output = path.parent / "evaluation" / f"{checkpoint}-{split}-e{state['epoch']}"
    output.mkdir(parents=True, exist_ok=False)
    records = dataset.rows[["sample_id", "target", "age", "race"]].copy()
    records["predicted"] = probabilities.argmax(1)
    records["predicted_age"] = np.asarray(AGE_CLASSES)[records.predicted]
    records[[f"probability_{i}" for i in range(9)]] = probabilities
    records.to_csv(output / "predictions.csv", index=False)
    summary = {"run": run_id, "experiment": saved["experiment"], "epoch": state["epoch"],
               "checkpoint": checkpoint, "split": split, "count": len(records),
               "smoke_subset": saved.get("limit") is not None, "precision": "fp32",
               "config": saved, **metrics, **scientific_metrics(records.target, records.predicted),
               **confidence_metrics(records.target, probabilities)}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_analysis(records, output)
    history = pd.read_csv(path.parent / "history.csv").assign(run=run_id)
    save_evaluation_figures(output, history)
    return {"output": str(output), **scientific_metrics(records.target, records.predicted)}


def compare_results(config, run_ids, checkpoint="best", split="development", name="comparison"):
    """Export derived combined tables and figures; original evaluations stay authoritative."""

    import matplotlib.pyplot as plt
    from fairage import plots

    tables = load_results(config, run_ids, checkpoint, split)
    output = Path(config["results_root"]) / name
    output.mkdir(parents=True, exist_ok=False)
    tables["metrics"].to_csv(output / "comparison.csv", index=False)
    for key in ("history", "by_age", "by_race", "by_age_race"):
        tables[key].to_csv(output / f"{key}.csv", index=False)
    for figure, _ in (
        plots.plot_learning_curves(tables["history"], save=True, save_path=output / "learning_curves.png"),
        plots.plot_loss_curves(tables["history"], save=True, save_path=output / "loss_curves.png"),
        plots.plot_training_efficiency(tables["history"], save=True, save_path=output / "epoch_timing.png"),
        plots.plot_model_comparison(tables["metrics"], save=True, save_path=output / "model_comparison.png"),
        plots.plot_age_f1(tables["by_age"], save=True, save_path=output / "per_age_comparison.png"),
        plots.plot_demographic_accuracy(tables["by_race"], save=True, save_path=output / "demographic_comparison.png"),
    ):
        plt.close(figure)
    return {"output": str(output)}
