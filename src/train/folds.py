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

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.splits import Fold, clips_from_index, fold_summary, folds_for_index
from src.features.source import FeatureSource
from src.results import fold_summary_path

logger = logging.getLogger(__name__)


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

        A fresh seed does not always buy a fresh split. ``StratifiedGroupKFold`` places
        groups greedily by class count, so with few enough groups every seed returns the
        same partition and the repeats vary only the model. That happened here and went
        unnoticed: the place grouping has 24 groups and produced one partition from ten
        repeats, so a score printed as fifty splits rested on five, and the fold count
        the paired test divides by was ten times too large. It is counted and said out
        loud rather than assumed.
        """

        def build(repeat: int) -> list[Fold]:
            shifted = replace(cfg, split=replace(cfg.split, seed=cfg.split.seed + repeat))
            return folds_for_index(index, shifted)

        plan = cls(build=build, repeats=repeats)
        plan.warn_if_repeats_buy_nothing()
        return plan

    def distinct_partitions(self) -> int:
        """How many different splits the repeats actually produce."""
        return len({tuple(tuple(sorted(fold.test_clips)) for fold in folds) for _, folds in self})

    def warn_if_repeats_buy_nothing(self) -> None:
        """Say so when the seed moved and the split did not.

        Silence here is the expensive failure. Ten repeats of an unchanging split cost
        ten times the compute, report ten times the folds, and add no evidence at all.
        """
        if self.repeats < 2:
            return
        distinct = self.distinct_partitions()
        if distinct == self.repeats:
            return
        logger.warning(
            "%d repeats produced only %d distinct partitions; the grouping has too few "
            "groups for a fresh seed to move the split, so these repeats vary the model "
            "and not the data. Read the fold count as %d rather than %d.",
            self.repeats,
            distinct,
            distinct * len(self.build(0)),
            self.repeats * len(self.build(0)),
        )


def folds_for(cfg: Config, source: FeatureSource) -> list[Fold]:
    """Build the folds a run will use.

    Folds come from the clips that actually produced features, so a clip that failed
    to decode cannot end up in a fold with no rows behind it.
    """
    return folds_for_index(source.index, cfg)


def _clips(cfg: Config, source: FeatureSource) -> pd.DataFrame:
    return clips_from_index(source.index, cfg.split.group_column, cfg)


def save_summary(cfg: Config, source: FeatureSource, folds: list[Fold]) -> Path:
    """Write clip and group counts per fold, per class, per part."""
    summary = fold_summary(_clips(cfg, source), folds, cfg.split.group_column)
    path = fold_summary_path(cfg)
    summary.to_csv(path, index=False)
    return path


def format_test_groups(cfg: Config, source: FeatureSource, folds: list[Fold]) -> str:
    """Independent units per class in each test fold, under whatever the fold rule is.

    This is the table that shows how thin the scarcest class really is; it belongs
    in front of anyone reading a score. It counts groups rather than tapes, because
    the group is what a held out fold actually holds out.
    """
    summary = fold_summary(_clips(cfg, source), folds, cfg.split.group_column)
    held_out = summary[summary["part"] == "test"]
    pivot = held_out.pivot(index="fold", columns="species", values="groups")
    return pivot.to_string()
