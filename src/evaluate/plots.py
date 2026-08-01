"""Figures.

Every figure also has a CSV behind it under the same report directory, so nothing
here is the only way to read a result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots are assigned to species in a fixed order and never cycled, so a
# species keeps its colour across every figure in the report.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

SEQUENTIAL = LinearSegmentedColormap.from_list("datura_blue", ["#f4f7fc", "#2a78d6", "#12365f"])


def species_colors(names: list[str]) -> dict[str, str]:
    return {name: SERIES[i % len(SERIES)] for i, name in enumerate(names)}


def _style(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor(SURFACE)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path


def sample_rate_profile(rates: pd.DataFrame, path: Path, target_rate: int) -> Path:
    """Which native sample rates each species was recorded at.

    The single most important figure in the audit. Where the rates separate by
    species, an unfiltered classifier can read the label off the recording
    equipment, and the cut line shows what the common-band filter removes.
    """
    species = list(dict.fromkeys(rates["species"]))
    colors = species_colors(species)
    figure, ax = plt.subplots(figsize=(8.5, 0.75 * len(species) + 2.2))

    for position, name in enumerate(species):
        # Rows run top to bottom in the order the config lists the species.
        row = len(species) - 1 - position
        subset = rates[rates["species"] == name].sort_values("clips")
        ax.scatter(
            subset["native_sample_rate"],
            np.full(len(subset), row),
            s=30 + 300 * subset["clips"] / rates["clips"].max(),
            color=colors[name],
            alpha=0.85,
            edgecolor=SURFACE,
            linewidth=1.4,
            zorder=3,
        )
        # Only the dominant rate is labelled. Several rates sit close together on a
        # log axis and a number on every point collides into noise.
        largest = subset.iloc[-1]
        ax.annotate(
            f"{int(largest['clips'])} clips at {int(largest['native_sample_rate'])} Hz",
            (largest["native_sample_rate"], row),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=8,
            color=MUTED,
        )

    ax.axvline(target_rate, color=INK, linewidth=1.4, linestyle="--", zorder=2)
    ax.annotate(
        f"kept at or above {target_rate} Hz",
        xy=(target_rate, 0.0),
        xycoords=("data", "axes fraction"),
        xytext=(6, 8),
        textcoords="offset points",
        color=INK,
        fontsize=8.5,
    )

    ax.set_xscale("log")
    ax.set_yticks(range(len(species)), species[::-1], color=INK)
    ax.set_ylim(-0.6, len(species) - 0.15)
    _style(ax, "Native sample rate by species, point size is clip count", "native sample rate (Hz)")
    ax.grid(axis="y", visible=False)
    return _save(figure, path)


def model_comparison(table: pd.DataFrame, path: Path, metric: str = "macro_f1") -> Path:
    """Mean and spread per model, with the control drawn as the reference line.

    The bar heights are less important than the gap between each audio model and the
    metadata control, so the control is also drawn across the whole axis.
    """
    ordered = table.sort_values("mean")
    figure, ax = plt.subplots(figsize=(7, 0.7 * len(ordered) + 1.6))

    control = ordered[ordered["model"] == "metadata"]
    colors = ["#8f8e88" if name == "metadata" else SERIES[0] for name in ordered["model"]]
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

    if not control.empty:
        floor = float(control["mean"].iloc[0])
        ax.axvline(floor, color="#8f8e88", linewidth=1.5, linestyle="--")
        ax.annotate(
            "metadata floor",
            xy=(floor, 1.0),
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
    """Row-normalised, so each row reads as recall for that species."""
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


def ambiguity_comparison(table: pd.DataFrame, path: Path) -> Path:
    """Each model scored on clips whose sample rate does and does not give the
    species away. The right-hand group is the one worth reading."""
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
    _style(
        ax, "Macro-F1 split by whether the sample rate identifies the species", ylabel="macro-F1"
    )
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


def occlusion_profile(table: pd.DataFrame, path: Path, class_names: list[str]) -> Path:
    """Macro-F1 lost when each frequency band is hidden from the trained model.

    This is the quantitative half of the explainability question. A band the model
    genuinely relies on costs accuracy when it disappears.
    """
    colors = species_colors(class_names)
    figure, ax = plt.subplots(figsize=(8, 4.2))
    centres = table["band_center_hz"].to_numpy()

    ax.plot(
        centres,
        table["macro_f1_drop"],
        color=INK,
        linewidth=2.0,
        marker="o",
        markersize=5,
        label="macro-F1",
        zorder=3,
    )
    for name in class_names:
        column = f"recall_{name}_drop"
        if column in table.columns:
            ax.plot(
                centres,
                table[column],
                color=colors[name],
                linewidth=1.6,
                marker="o",
                markersize=4,
                label=name,
                alpha=0.9,
            )

    ax.axhline(0, color=GRID, linewidth=1.0)
    _style(ax, "Score lost when a frequency band is masked", "band centre (Hz)", "drop")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(figure, path)


def gradcam_panel(
    spectrograms: np.ndarray,
    heatmaps: np.ndarray,
    labels: list[str],
    path: Path,
    frequencies: np.ndarray | None = None,
    seconds: float | None = None,
) -> Path:
    """Windows with the model's attention drawn over them."""
    n = len(spectrograms)
    columns = min(n, 3)
    rows = int(np.ceil(n / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.6 * rows), squeeze=False)

    extent = None
    if frequencies is not None and seconds is not None:
        extent = (0.0, seconds, float(frequencies[0]), float(frequencies[-1]))

    for position, ax in enumerate(axes.flat):
        if position >= n:
            ax.axis("off")
            continue
        heat = heatmaps[position]
        ax.imshow(
            spectrograms[position], origin="lower", aspect="auto", cmap="gray_r", extent=extent
        )
        # Alpha tracks the map itself, so the spectrogram stays readable wherever the
        # model was not looking instead of being washed out by a flat overlay.
        ax.imshow(
            heat,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            alpha=np.clip(heat, 0.0, 1.0) * 0.8,
            extent=extent,
            vmin=0,
            vmax=1,
        )
        ax.set_title(labels[position], color=INK, fontsize=9, loc="left")
        ax.tick_params(colors=MUTED, labelsize=8, length=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if extent:
            ax.set_xlabel("s", color=MUTED, fontsize=8)
            ax.set_ylabel("Hz", color=MUTED, fontsize=8)

    figure.suptitle("Grad-CAM over log-mel windows", color=INK, fontsize=11, x=0.02, ha="left")
    figure.tight_layout()
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
