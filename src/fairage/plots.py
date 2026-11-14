"""Draw scientific figures from saved tables and arrays without inference.

Main functions:
    plot_learning_curves: Compare recorded training/development trajectories.
    plot_model_comparison: Compare aggregate classification and ordinal scores.
    plot_age_metrics: Show per-age precision, recall, F1, and support.
    plot_age_race_support: Show annotated age-by-race support.

Each plot returns (figure, axes). Saving requires save=True and save_path;
the caller controls display, closing, and the Matplotlib backend.
Headless callers may set MPLBACKEND=Agg before importing pyplot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_figure(figure, save, save_path):
    """Save a 300-DPI PNG at save_path only when save is true.

    Require an explicit path, create its parents, and leave the figure open.
    Existing files at that path can be replaced; the caller chooses the path.
    """

    if not save:
        return
    if save_path is None:
        raise ValueError("save_path is required when save=True")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", dpi=300, bbox_inches="tight")


def plot_learning_curves(data, save: bool = False, save_path: str | Path | None = None):
    """Plot recorded train/development F1 and accuracy; stars mark best dev-F1 epochs."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for run_id, history in data.groupby("run", sort=False):
        best = history.loc[history.dev_macro_f1.idxmax()]
        for axis, metric, label in zip(axes, ("macro_f1", "accuracy"), ("Nine-class macro-F1", "Accuracy")):
            line, = axis.plot(history.epoch, history[f"train_{metric}"], label=run_id)
            axis.plot(history.epoch, history[f"dev_{metric}"], "--", color=line.get_color())
            axis.scatter(best.epoch, best[f"dev_{metric}"], marker="*", s=95,
                         color=line.get_color(), zorder=3)
            axis.set(xlabel="Completed epoch", ylabel=label)
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Solid: training; dashed: development; stars: best development-F1 epoch")
    figure.tight_layout()
    save_figure(figure, save, save_path)
    return figure, axes


def plot_loss_curves(data, save: bool = False, save_path: str | Path | None = None):
    """Compare recorded training/development cross-entropy losses across runs."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for run_id, history in data.groupby("run", sort=False):
        line, = axis.plot(history.epoch, history.train_loss, label=run_id)
        axis.plot(history.epoch, history.dev_loss, "--", color=line.get_color())
    axis.set(xlabel="Completed epoch", ylabel="Cross-entropy loss",
             title="Loss: solid training, dashed development")
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    figure.tight_layout()
    save_figure(figure, save, save_path)
    return figure, axis


def plot_training_efficiency(data, save: bool = False, save_path: str | Path | None = None):
    """Show existing per-epoch train/dev seconds; no total-runtime accounting."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for run_id, history in data.groupby("run", sort=False):
        line, = axis.plot(history.epoch, history.train_seconds, label=run_id)
        axis.plot(history.epoch, history.dev_seconds, "--", color=line.get_color())
    axis.set(xlabel="Completed epoch", ylabel="Recorded seconds per epoch phase",
             title="Per-epoch timing: solid training, dashed development")
    axis.legend(fontsize=8)
    figure.tight_layout()
    save_figure(figure, save, save_path)
    return figure, axis


def plot_confusion_matrix(matrix, normalized=False, save: bool = False,
                          save_path: str | Path | None = None):
    """Plot a labeled count matrix, optionally normalized by true-age support."""

    counts = np.asarray(matrix)
    labels = list(matrix.index) if hasattr(matrix, "index") else list(range(len(counts)))
    values = counts
    if normalized:
        support = counts.sum(axis=1, keepdims=True)
        values = np.divide(counts, support, out=np.zeros_like(counts, dtype=float), where=support != 0)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else max(1, counts.max()))
    axis.set(xticks=range(len(counts)), yticks=range(len(counts)), xticklabels=labels,
             yticklabels=labels, xlabel="Predicted age group", ylabel="Annotated age group",
             title="Confusion matrix: row fractions" if normalized else "Confusion matrix: counts")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(counts)):
        for column in range(len(counts)):
            value = values[row, column]
            axis.text(column, row, f"{value:.2f}" if normalized else str(value), ha="center", va="center",
                      fontsize=8, color="white" if value > image.norm.vmax / 2 else "black")
    figure.colorbar(image, ax=axis, label="Fraction" if normalized else "Images")
    figure.tight_layout()
    save_figure(figure, save, save_path)
    return figure, axis


def plot_age_metrics(data, save: bool = False, save_path: str | Path | None = None):
    """Show precision/recall/F1 and annotated support for one evaluated model."""

    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(data))
    for offset, metric in enumerate(("precision", "recall", "f1")):
        axis.bar(positions + (offset - 1) * .25, data[metric], width=.25, label=metric)
    labels = [f"{row.age}\n(n={row.support:,})" for row in data.itertuples()]
    axis.set(xticks=positions, xticklabels=labels, ylim=(0, 1), ylabel="Score",
             xlabel="Annotated age group", title="Per-age classification metrics")
    axis.legend()
    figure.tight_layout()
    save_figure(figure, save, save_path)
    return figure, axis


def plot_age_f1(data, save: bool = False, save_path: str | Path | None = None):
    """Compare F1 across the nine annotated age categories using saved per-age rows."""

    figure, axis = plt.subplots(figsize=(11, 5))
    labels = data.age.drop_duplicates().tolist()
    for run_id, rows in data.groupby("run", sort=False):
        axis.plot(range(len(labels)), rows.set_index("age").reindex(labels).f1, marker="o", label=run_id)
    axis.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, 1), ylabel="F1",
             title="Nine-class per-age F1 comparison")
    axis.legend(fontsize=8)
    figure.tight_layout()
    save_figure(figure, save, save_path)
    return figure, axis


def plot_demographic_accuracy(data, save: bool = False, save_path: str | Path | None = None):
    """Compare annotated-race accuracy/F1 with support; empty groups retain no score."""

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = data.race.drop_duplicates().tolist()
    for run_id, rows in data.groupby("run", sort=False):
        rows = rows.set_index("race").reindex(labels)
        for axis, metric in zip(axes, ("accuracy", "macro_f1", "support")):
            axis.plot(range(len(labels)), rows[metric], marker="o", label=run_id)
            axis.set(ylabel="Images (support)" if metric == "support" else metric.replace("_", " "),
                     xticks=range(len(labels)), xticklabels=labels,
                     ylim=(0, None) if metric == "support" else (0, 1))
            plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    axes[0].legend(fontsize=8)
    figure.suptitle("Annotated-race comparison (empty groups have no score)")
    figure.tight_layout()
    save_figure(figure, save, save_path)
    return figure, axes


def plot_age_race_support(data, save: bool = False, save_path: str | Path | None = None):
    """Plot released age-by-race counts; missing counts are masked, never zero-filled.

    The public suppressed flag concerns accuracy, not these support counts.
    """

    ages, races = data.age.drop_duplicates().tolist(), data.race.drop_duplicates().tolist()
    matrix = data.pivot(index="age", columns="race", values="support").reindex(index=ages, columns=races)
    counts = matrix.to_numpy(dtype=float)
    visible = np.isfinite(counts)
    color_map = plt.get_cmap("Blues").with_extremes(bad="#ECEFF1")
    figure, axis = plt.subplots(figsize=(11, 7))
    image = axis.imshow(np.ma.masked_invalid(counts), cmap=color_map,
                        vmin=0, vmax=max(1, counts[visible].max()) if visible.any() else 1)
    axis.set(xticks=range(len(races)), xticklabels=races, yticks=range(len(ages)), yticklabels=ages,
             ylabel="Annotated age group", title="Age-by-race support (annotated groups)")
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    for row in range(len(ages)):
        for column in range(len(races)):
            value = counts[row, column]
            label = f"{int(value):,}" if visible[row, column] else "withheld"
            axis.text(column, row, label, ha="center", va="center", fontsize=8,
                      color="white" if visible[row, column] and value > image.norm.vmax / 2 else "black")
    figure.colorbar(image, ax=axis, label="Images")
    figure.tight_layout()
    save_figure(figure, save, save_path)
    return figure, axis


def plot_model_comparison(data, save: bool = False, save_path: str | Path | None = None):
    """Compare saved overall scores and ordinal error on separate appropriate scales."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    positions = np.arange(len(data))
    for offset, metric in enumerate(("accuracy", "macro_f1", "balanced_accuracy", "within_one_bin_accuracy")):
        axes[0].bar(positions + (offset - 1.5) * .18, data[metric], width=.18, label=metric.replace("_", " "))
    axes[0].set(ylim=(0, 1), ylabel="Score", title="Overall evaluation")
    axes[0].legend(fontsize=8)
    axes[1].bar(positions, data.mean_absolute_bin_index_error)
    axes[1].set(ylabel="Mean absolute bin-index error", title="Ordinal error (not years)")
    for axis in axes:
        axis.set(xticks=positions, xticklabels=data.run)
        plt.setp(axis.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    figure.tight_layout()
    save_figure(figure, save, save_path)
    return figure, axes


# Publisher labels stay unchanged in tables; these are display-only aliases.
DISPLAY_LABELS = {"more than 70": "Over 70", "Latino_Hispanic": "Latino/Hispanic"}
