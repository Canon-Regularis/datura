"""Figures showing what a trained network relied on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.evaluate.plots.style import (
    GRID,
    INK,
    MUTED,
    _save,
    _style,
    species_colors,
)


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

    figure.suptitle("Grad-CAM over log mel windows", color=INK, fontsize=11, x=0.02, ha="left")
    figure.tight_layout()
    return _save(figure, path)
