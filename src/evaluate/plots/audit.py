"""Figures describing the dataset, before any model has seen it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.evaluate.plots.style import (
    INK,
    MUTED,
    SURFACE,
    _save,
    _style,
    species_colors,
)


def sample_rate_profile(rates: pd.DataFrame, path: Path, target_rate: int) -> Path:
    """Which native sample rates each species was recorded at.

    The single most important figure in the audit. Where the rates separate by
    species, an unfiltered classifier can read the label off the recording
    equipment, and the cut line shows what the common band filter removes.
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
