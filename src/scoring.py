"""Scoring, shared by every model so the numbers stay comparable.

This sits below both training and evaluation because both need it. The cross
validation runner scores each fold as it fits; the report scores subsets of the
same predictions afterwards. Neither owns it.

Two things here matter more than the metric list.

Predictions are aggregated from windows to clips before scoring. Windows cut from
one clip are not independent observations, so a window level score counts the same
recording many times over.

Results are summarised as a mean and a spread across folds. Humpback whale survives
on roughly a dozen tapes, so a single figure would hide most of the uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def probability_columns(n_classes: int) -> list[str]:
    """Column names holding per class scores in an aggregated frame.

    Four call sites need this convention. Naming it once means a change to the
    layout cannot leave one of them reading the wrong columns.
    """
    return [f"p{i}" for i in range(n_classes)]


def aggregate_to_clips(
    index: pd.DataFrame, rows: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    """Average window probabilities within each clip.

    Returns one row per clip with its true label and the averaged class scores.
    """
    if len(rows) != len(probabilities):
        raise ValueError(f"{len(rows)} rows but {len(probabilities)} probability vectors")

    subset = index.iloc[rows].reset_index(drop=True)
    scores = pd.DataFrame(probabilities, columns=probability_columns(probabilities.shape[1]))
    joined = pd.concat([subset[["clip_id", "tape_id", "species", "label"]], scores], axis=1)

    score_columns = list(scores.columns)
    grouped = joined.groupby("clip_id", as_index=False).agg(
        {
            "tape_id": "first",
            "species": "first",
            "label": "first",
            **{column: "mean" for column in score_columns},
        }
    )
    grouped["prediction"] = grouped[score_columns].to_numpy().argmax(axis=1)
    return grouped


def evaluate_clips(
    index: pd.DataFrame,
    rows: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Aggregate window predictions to clips, then score them.

    Every model is judged through this one path: the cross validation runner and the
    occlusion test both call it, so a change to how clips are scored cannot apply to
    one and not the other.
    """
    clips = aggregate_to_clips(index, rows, probabilities)
    clip_probabilities = clips[probability_columns(len(class_names))].to_numpy()
    return clips, score(clips["label"].to_numpy(), clip_probabilities, class_names)


def _safe_roc_auc(labels: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    """One against the rest AUC over the classes that appear, or NaN when it is undefined.

    A class with no examples in a fold has no ROC curve, so its columns are dropped
    and the remainder renormalised rather than scoring an empty class as chance.
    """
    present = np.unique(labels)
    if len(present) < 2:
        return float("nan")

    columns = probabilities[:, present]
    totals = columns.sum(axis=1, keepdims=True)
    normalised = np.divide(
        columns, totals, out=np.full_like(columns, 1.0 / len(present)), where=totals > 0
    )
    try:
        if len(present) == 2:
            return float(roc_auc_score((labels == present[1]).astype(int), normalised[:, 1]))
        if len(present) < n_classes:
            return float(
                roc_auc_score(
                    labels, normalised, multi_class="ovr", average="macro", labels=present
                )
            )
        return float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _safe_average_precision(labels: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    one_hot = np.zeros((len(labels), n_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    present = one_hot.sum(axis=0) > 0
    if present.sum() < 2:
        return float("nan")
    try:
        return float(
            average_precision_score(one_hot[:, present], probabilities[:, present], average="macro")
        )
    except ValueError:
        return float("nan")


COUNT_METRICS = ("accuracy", "macro_f1", "weighted_f1")


def from_counts(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> dict[str, float]:
    """Accuracy and the F1 averages, from the confusion counts alone.

    ``score`` also computes ranking metrics, and those cost more than everything
    else together. The bootstrap rescores thousands of times and asks for one number,
    so it needs the cheap path. ``score`` is built on the same function rather than
    on its own copy, because two implementations of macro F1 would eventually
    disagree and only one of them would be in the report.

    A class with no support and no predictions scores zero rather than dividing by
    nothing, which is what sklearn does with ``zero_division=0``.
    """
    matrix = np.bincount(
        labels.astype(np.int64) * n_classes + predictions.astype(np.int64),
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)

    correct = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    actual = matrix.sum(axis=1).astype(float)

    denominator = predicted + actual
    per_class = np.divide(
        2.0 * correct, denominator, out=np.zeros(n_classes), where=denominator > 0
    )

    total = float(actual.sum())
    if total == 0.0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    return {
        "accuracy": float(correct.sum() / total),
        "macro_f1": float(per_class.mean()),
        "weighted_f1": float(np.dot(per_class, actual) / total),
    }


def score(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Headline metrics for one set of predictions."""
    n_classes = len(class_names)
    all_labels = list(range(n_classes))
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=all_labels, zero_division=0
    )

    # The full class set is passed everywhere, so the averaged figures and the
    # per class table always describe the same classes. Without it a fold missing a
    # species would report a macro average over the survivors while the table below
    # still listed the absent one at zero.
    result: dict[str, float] = {
        **from_counts(labels, predictions, n_classes),
        "roc_auc_ovr_macro": _safe_roc_auc(labels, probabilities, n_classes),
        "average_precision_macro": _safe_average_precision(labels, probabilities, n_classes),
        "n_items": float(len(labels)),
    }
    for i, name in enumerate(class_names):
        result[f"precision_{name}"] = float(precision[i])
        result[f"recall_{name}"] = float(recall[i])
        result[f"f1_{name}"] = float(f1[i])
        result[f"support_{name}"] = float(support[i])
    return result


def confusion(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    return pd.DataFrame(matrix, index=class_names, columns=class_names)


def summarise_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of every metric across folds and repeats.

    A repeated run has as many rows as repeats times folds, and every one of them is
    an estimate of the same quantity, so they are summarised together.
    """
    numeric = fold_metrics.drop(columns=["fold", "repeat"], errors="ignore").select_dtypes("number")
    summary = pd.DataFrame(
        {
            "metric": numeric.columns,
            "mean": numeric.mean().to_numpy(),
            "std": numeric.std(ddof=1).to_numpy(),
            "min": numeric.min().to_numpy(),
            "max": numeric.max().to_numpy(),
        }
    )
    return summary.reset_index(drop=True)


def format_headline(model_name: str, summary: pd.DataFrame) -> str:
    """One line per model, always carrying the spread alongside the mean."""
    lookup = summary.set_index("metric")
    parts = []
    for metric, label in (("accuracy", "acc"), ("macro_f1", "macro-F1")):
        if metric in lookup.index:
            row = lookup.loc[metric]
            parts.append(f"{label} {row['mean']:.3f} +/- {row['std']:.3f}")
    return f"{model_name:>10}: " + "  ".join(parts)
