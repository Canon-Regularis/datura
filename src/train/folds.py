"""Fold assembly for the training entry points.

``src.data.splits`` owns the grouping rule. This module puts it to work: it turns a
window index into folds, saves the shape of those folds, and formats them for
reading. Those are three separate jobs, so they are three separate functions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.splits import Fold, clips_from_index, fold_summary, folds_for_index
from src.features.source import FeatureSource
from src.results import fold_summary_path


def folds_for(cfg: Config, source: FeatureSource) -> list[Fold]:
    """Build the folds a run will use.

    Folds come from the clips that actually produced features, so a clip that failed
    to decode cannot end up in a fold with no rows behind it.
    """
    return folds_for_index(source.index, cfg)


def save_summary(cfg: Config, source: FeatureSource, folds: list[Fold]) -> Path:
    """Write clip and tape counts per fold, per species, per part."""
    summary = fold_summary(clips_from_index(source.index), folds)
    path = fold_summary_path(cfg)
    summary.to_csv(path, index=False)
    return path


def format_test_tapes(cfg: Config, source: FeatureSource, folds: list[Fold]) -> str:
    """Tapes per species in each test fold.

    This is the table that shows how thin the scarcest class really is; it belongs
    in front of anyone reading a score.
    """
    summary = fold_summary(clips_from_index(source.index), folds)
    held_out = summary[summary["part"] == "test"]
    pivot = held_out.pivot(index="fold", columns="species", values="tapes")
    return pivot.to_string()


def describe(cfg: Config, source: FeatureSource, folds: list[Fold]) -> tuple[list[Fold], str]:
    """Build folds, save their shape, and return them with a table to log."""
    save_summary(cfg, source, folds)
    return folds, format_test_tapes(cfg, source, folds)


def clip_counts(source: FeatureSource) -> pd.DataFrame:
    """Clips and tapes per species behind the current feature cache."""
    clips = clips_from_index(source.index)
    return (
        clips.groupby("species")
        .agg(clips=("clip_id", "size"), tapes=("tape_id", "nunique"))
        .reset_index()
    )
