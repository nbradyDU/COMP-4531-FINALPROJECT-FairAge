"""Calculate shared classification and ordinal-bin metrics for FairAge.

Use NumPy and scikit-learn to summarize aligned age-category predictions.
Metric definitions are shared by training and evaluation.

Main functions:
    classification_metrics: Return accuracy and fixed-nine macro-F1.
    scientific_metrics: Add balanced accuracy and ordinal-bin error summaries.
    confidence_metrics: Summarize confidence, calibration and confident errors.
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_metrics(actual, predicted):
    """Use accuracy and macro-F1 across all nine classes, including absent classes."""

    return {"accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=range(9),
                                      average="macro", zero_division=0))}


def scientific_metrics(actual, predicted):
    """Summarize aligned one-dimensional class-index arrays in fixed age order.

    Macro-F1 includes all nine classes (undefined F1 is zero); balanced accuracy
    averages recall over true classes present in this subset. Ordinal distances
    use indices 0..8, not chronological years. Empty groups return missing scores.
    """

    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) == 0:
        return {key: None for key in ("accuracy", "macro_f1", "balanced_accuracy",
                "within_one_bin_accuracy", "mean_absolute_bin_index_error")}
    distances = abs(actual - predicted)
    return {**classification_metrics(actual, predicted),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "within_one_bin_accuracy": float((distances <= 1).mean()),
            "mean_absolute_bin_index_error": float(distances.mean())}


def confidence_metrics(actual, probabilities):
    """Summarize nonempty [N,9] probabilities using the publication definitions.

    Confidence is the maximum class probability. ECE uses 10/15 equal-width
    [0,1] bins, left-inclusive/right-exclusive except that the final bin
    includes 1. Empty bins contribute zero to the sample-weighted absolute
    confidence/accuracy gap. Wrong predictions above 90% use strictly > .9;
    wrong_count is the denominator for reporting their fraction.

    Float64 matches confidence_diagnostics in the publication source at 8f11582.
    This calculation does not change probabilities or checkpoint selection.
    """

    actual, probabilities = np.asarray(actual), np.asarray(probabilities, dtype=float)
    if actual.ndim != 1 or not len(actual) or probabilities.shape != (len(actual), 9):
        raise ValueError("Nonempty aligned targets and [N,9] probabilities are required")
    if (not np.isin(actual, range(9)).all() or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError("Use age indices 0..8 and normalized finite probabilities in [0,1]")
    confidence = probabilities.max(axis=1)
    correct = actual == probabilities.argmax(axis=1)
    summary = {"mean_max_probability": float(confidence.mean()),
               "wrong_count": int((~correct).sum()),
               "wrong_above_90": int(((~correct) & (confidence > .9)).sum())}
    for bins in (10, 15):
        ece = 0.0
        for index in range(bins):
            lower, upper = index / bins, (index + 1) / bins
            selected = (confidence >= lower) & ((confidence < upper) if index < bins - 1
                                              else (confidence <= 1))
            count = int(selected.sum())
            if count:
                ece += count / len(actual) * abs(confidence[selected].mean() - correct[selected].mean())
        summary[f"ece_{bins}"] = float(ece)
    return summary
