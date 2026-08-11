"""Figures comparing what the models achieved."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.evaluate.plots.style import (
    INK,
    MUTED,
    SEQUENTIAL,
    SERIES,
    SURFACE,
    _save,
    _style,
    species_colors,
)


def model_comparison(
    table: pd.DataFrame,
    path: Path,
    floor: str,
    silent: set[str] | None = None,
    metric: str = "macro_f1",
) -> Path:
    """Mean and spread per model, with the floor drawn as the reference line.

    The bar heights are less important than the gap between each audio model and the
    highest score reached without hearing the recording, so that model is also drawn
    across the whole axis.

    ``floor`` is named by the caller rather than found here. This chart drew its line
    at the metadata control for as long as the control was assumed to be the floor,
    which put it a tenth of a point below where an audio model actually had to reach.
    ``silent`` names every model that hears no audio, so all of them are greyed rather
    than the one that happens to be called metadata.
    """
    ordered = table.sort_values("mean")
    figure, ax = plt.subplots(figsize=(7, 0.7 * len(ordered) + 1.6))

    silent = silent or {floor}
    reference = ordered[ordered["model"] == floor]
    colors = ["#8f8e88" if name in silent else SERIES[0] for name in ordered["model"]]
    positions = np.arange(len(ordered))

    ax.barh(
        positions,
        ordered["mean"],
        xerr=ordered["std"],
        color=colors,
        height=0.55,
        error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capsize": 4},
    )
    for y, (mean, std) in enumerate(zip(ordered["mean"], ordered["std"], strict=True)):
        ax.text(
            mean + std + 0.015, y, f"{mean:.3f} ± {std:.3f}", va="center", color=INK, fontsize=9
        )

    if not reference.empty:
        height = float(reference["mean"].iloc[0])
        ax.axvline(height, color="#8f8e88", linewidth=1.5, linestyle="--")
        ax.annotate(
            f"{floor} floor",
            xy=(height, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(5, -6),
            textcoords="offset points",
            color=MUTED,
            fontsize=8,
            va="top",
        )

    ax.set_yticks(positions, ordered["model"], color=INK)
    ax.set_xlim(0, 1.18)
    _style(
        ax,
        f"{metric.replace('_', ' ')} by model, mean and spread over folds",
        metric.replace("_", " "),
    )
    ax.grid(axis="y", visible=False)
    return _save(figure, path)


def confusion_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> Path:
    """Row normalised, so each row reads as recall for that species."""
    counts = matrix.to_numpy(dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    figure, ax = plt.subplots(figsize=(1.35 * len(matrix) + 2.6, 1.2 * len(matrix) + 2.0))
    ax.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1)

    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            ax.text(
                j,
                i,
                f"{shares[i, j]:.2f}\n{int(counts[i, j])}",
                ha="center",
                va="center",
                fontsize=9,
                color="#ffffff" if shares[i, j] > 0.55 else INK,
            )

    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("predicted", color=MUTED, fontsize=9)
    ax.set_ylabel("actual", color=MUTED, fontsize=9)
    return _save(figure, path)


def per_class_recall(table: pd.DataFrame, path: Path, class_names: list[str]) -> Path:
    """Recall per species per model. The scarcest class is the one that matters."""
    colors = species_colors(class_names)
    figure, ax = plt.subplots(figsize=(7.5, 4.0))
    models = list(table["model"])
    positions = np.arange(len(models))
    width = 0.8 / len(class_names)

    for i, name in enumerate(class_names):
        means = table[f"recall_{name}_mean"].to_numpy()
        errors = table[f"recall_{name}_std"].to_numpy()
        ax.bar(
            positions + i * width - 0.4 + width / 2,
            means,
            width=width * 0.88,
            yerr=errors,
            label=name,
            color=colors[name],
            error_kw={"ecolor": MUTED, "elinewidth": 1.0, "capsize": 3},
        )

    ax.set_xticks(positions, models, color=INK)
    ax.set_ylim(0, 1.05)
    _style(ax, "Recall per species, mean and spread over folds", ylabel="recall")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncols=len(class_names))
    return _save(figure, path)


def coverage_curve(table: pd.DataFrame, path: Path) -> Path:
    """What each model is worth when it is allowed to decline.

    Every other figure here forces an answer for every clip, which is the right way to
    compare two representations and the wrong way to describe a tool. This reads left
    to right as the model becoming choosier, and a curve that climbs is a model whose
    confidence carries information.

    A flat curve would be the interesting failure: a model equally wrong when sure and
    unsure, for which no threshold buys anything.
    """
    figure, ax = plt.subplots(figsize=(7.5, 4.2))
    names = list(dict.fromkeys(table["model"]))

    for index, name in enumerate(names):
        rows = table[table["model"] == name].sort_values("coverage")
        ax.plot(
            rows["coverage"] * 100,
            rows["accuracy"] * 100,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=SERIES[index % len(SERIES)],
            label=name,
        )

    ax.invert_xaxis()
    _style(
        ax,
        "Accuracy against the share of clips answered",
        xlabel="coverage, percent of clips the model answered",
        ylabel="accuracy on those clips, percent",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(figure, path)


def ambiguity_comparison(table: pd.DataFrame, path: Path, field: str) -> Path:
    """Each model scored on clips one field does and does not give the species away on.

    ``field`` is passed in because one call draws this per giveaway. The title used to
    name the sample rate whatever it had been handed, so the collection code figure
    carried a caption about equipment.
    """
    subsets = list(dict.fromkeys(table["subset"]))
    models = list(dict.fromkeys(table["model"]))
    figure, ax = plt.subplots(figsize=(7.5, 4.2))
    positions = np.arange(len(models))
    width = 0.8 / len(subsets)

    for i, subset in enumerate(subsets):
        rows = table[table["subset"] == subset].set_index("model").reindex(models)
        offsets = positions + i * width - 0.4 + width / 2
        ax.bar(
            offsets,
            rows["macro_f1_mean"],
            width=width * 0.88,
            yerr=rows["macro_f1_std"],
            label=subset,
            color=SERIES[i % len(SERIES)],
            error_kw={"ecolor": MUTED, "elinewidth": 1.0, "capsize": 3},
        )
        for x, clips in zip(offsets, rows["clips"], strict=True):
            ax.text(
                x,
                0.02,
                f"n={int(clips)}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=SURFACE,
                rotation=90,
            )

    ax.set_xticks(positions, models, color=INK)
    ax.set_ylim(0, 1.05)
    _style(ax, f"Macro-F1 split by what the {field} does to the species", ylabel="macro-F1")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(figure, path)


def feature_importance(table: pd.DataFrame, path: Path, top_n: int = 25) -> Path:
    aggregated = table.groupby("feature", as_index=False)["gain"].mean().nlargest(top_n, "gain")
    figure, ax = plt.subplots(figsize=(7, 0.26 * len(aggregated) + 1.4))
    positions = np.arange(len(aggregated))[::-1]
    ax.barh(positions, aggregated["gain"], color=SERIES[0], height=0.65)
    ax.set_yticks(positions, aggregated["feature"], fontsize=8, color=INK)
    _style(ax, f"Top {top_n} features by mean gain across folds", "gain")
    ax.grid(axis="y", visible=False)
    return _save(figure, path)


def training_history(table: pd.DataFrame, path: Path) -> Path:
    figure, ax = plt.subplots(figsize=(7, 4.0))
    for fold, group in table.groupby("fold"):
        ax.plot(
            group["epoch"],
            group["val_macro_f1"],
            linewidth=1.8,
            color=SERIES[int(fold) % len(SERIES)],
            label=f"fold {fold}",
        )
    _style(ax, "Validation macro-F1 per epoch", "epoch", "macro-F1")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncols=3)
    return _save(figure, path)
