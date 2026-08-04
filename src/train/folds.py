"""Fold assembly for the training entry points.

``src.data.splits`` owns the grouping rule. This module puts it to work: it turns a
window index into folds, saves the shape of those folds, and formats them for
reading. Those are three separate jobs, so they are three separate functions.

A run may also be scored on several different splits rather than one. Five folds
over a dozen recordings gives five estimates, which is too few to separate the
differences this project reports; repeating the whole split under fresh seeds gives
as many as the compute allows. ``FoldPlan`` carries that choice so the runner does
not have to know whether it is doing one split or ten.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.splits import Fold, clips_from_index, fold_summary, folds_for_index
from src.features.source import FeatureSource
from src.results import fold_summary_path


@dataclass(frozen=True)
class FoldPlan:
    """The splits a run will be scored on, and how many times over."""

    build: Callable[[int], list[Fold]]
    repeats: int = 1

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError("a fold plan needs at least one repeat")

    def __iter__(self) -> Iterator[tuple[int, list[Fold]]]:
        for repeat in range(self.repeats):
            yield repeat, self.build(repeat)

    @classmethod
    def single(cls, folds: list[Fold]) -> FoldPlan:
        """One split, already built. What every run did before repeats existed."""
        return cls(build=lambda _: folds, repeats=1)

    @classmethod
    def repeated(cls, cfg: Config, index: pd.DataFrame, repeats: int) -> FoldPlan:
        """Fresh folds per repeat, by moving the split seed.

        Repeat zero reproduces the single split exactly, so a repeated run contains
        the original one rather than replacing it.
        """

        def build(repeat: int) -> list[Fold]:
            shifted = replace(cfg, split=replace(cfg.split, seed=cfg.split.seed + repeat))
            return folds_for_index(index, shifted)

        return cls(build=build, repeats=repeats)


def folds_for(cfg: Config, source: FeatureSource) -> list[Fold]:
    """Build the folds a run will use.

    Folds come from the clips that actually produced features, so a clip that failed
    to decode cannot end up in a fold with no rows behind it.
    """
    return folds_for_index(source.index, cfg)


def _clips(cfg: Config, source: FeatureSource) -> pd.DataFrame:
    return clips_from_index(source.index, cfg.split.group_column)


def save_summary(cfg: Config, source: FeatureSource, folds: list[Fold]) -> Path:
    """Write clip and group counts per fold, per class, per part."""
    summary = fold_summary(_clips(cfg, source), folds, cfg.split.group_column)
    path = fold_summary_path(cfg)
    summary.to_csv(path, index=False)
    return path


def format_test_tapes(cfg: Config, source: FeatureSource, folds: list[Fold]) -> str:
    """Independent recordings per class in each test fold.

    This is the table that shows how thin the scarcest class really is; it belongs
    in front of anyone reading a score.
    """
    summary = fold_summary(_clips(cfg, source), folds, cfg.split.group_column)
    held_out = summary[summary["part"] == "test"]
    pivot = held_out.pivot(index="fold", columns="species", values="tapes")
    return pivot.to_string()
