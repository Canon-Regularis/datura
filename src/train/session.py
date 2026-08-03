"""Training one model, end to end.

Both entry points funnel through here. Loading the right features, building the
folds, running the cross validation and writing the results is the same work
whichever model is being trained, so it is written once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Config
from src.data.splits import Fold
from src.features import registry as features
from src.features.source import FeatureSource, MetadataFeatureSource
from src.models.registry import METADATA_SOURCE, ModelSpec
from src.train.crossval import run_cross_validation, save_result
from src.train.folds import FoldPlan, folds_for, format_test_tapes, save_summary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Assembly:
    """The feature sources and folds a run needs, built once and shared.

    Both tree models are scored on the same folds as the network. Building the
    folds here, from the window index, is what guarantees that.
    """

    audio: FeatureSource
    folds: list[Fold]
    plan: FoldPlan

    def source_for(self, spec: ModelSpec) -> FeatureSource:
        """The features a given model consumes.

        The control is derived from the audio source's index: it sees the same
        clips, in the same folds, described only by their recording metadata.
        """
        if spec.source == METADATA_SOURCE:
            return MetadataFeatureSource(self.audio.index)
        if spec.source != self.audio.name:
            raise ValueError(
                f"{spec.name} needs {spec.source} features; this assembly holds {self.audio.name}"
            )
        return self.audio


def assemble(cfg: Config, source_kind: str, repeats: int = 1) -> Assembly:
    """Open a feature cache, build the folds, and log the shape of both.

    ``repeats`` reruns the whole split under fresh seeds. Five folds over a dozen
    recordings cannot separate the differences this project reports, and repeating
    the split is the cheap way to get more estimates of the same quantity.
    """
    source = features.load_source(source_kind, cfg)
    folds = folds_for(cfg, source)
    save_summary(cfg, source, folds)

    logger.info(
        "%s features: %d windows over %d clips and %d tapes",
        source.name,
        len(source.index),
        source.index["clip_id"].nunique(),
        source.index[cfg.split.group_column].nunique(),
    )
    logger.info(
        "\nIndependent recordings per class in each test fold\n%s",
        format_test_tapes(cfg, source, folds),
    )
    if repeats > 1:
        logger.info("scoring on %d repeats of the split, %d folds each", repeats, len(folds))
    plan = FoldPlan.repeated(cfg, source.index, repeats) if repeats > 1 else FoldPlan.single(folds)
    return Assembly(audio=source, folds=folds, plan=plan)


def train(
    cfg: Config,
    spec: ModelSpec,
    assembly: Assembly,
    settings: dict[str, Any],
    name: str | None = None,
) -> Path:
    """Fit one model across every fold and write what it produced.

    ``name`` overrides the result directory, which is how two capacities of the same
    network are trained and reported side by side.
    """
    source = assembly.source_for(spec)
    result_name = name or spec.name
    logger.info("\n%s on %s features: %s", result_name, source.name, spec.summary)

    result = run_cross_validation(
        cfg,
        source,
        assembly.plan,
        lambda: spec.build(cfg, settings),
        result_name,
    )
    directory = save_result(cfg, result)
    logger.info("%s", result.headline())
    logger.info("  written to %s", directory)
    return directory
