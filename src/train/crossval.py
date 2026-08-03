"""The shared cross validation runner.

One loop fits every model in the project. It never learns what kind of model it is
holding, so the acoustic baseline, the network and the metadata control are
evaluated under identical folds, identical aggregation and identical scoring. The
harness is the same for all of them; any gap between their reported numbers came
from the features.

Per fold extras come from the model itself, through ``artifacts``. That is why
there is no branch here for feature importance or for learning curves.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import scoring
from src.config import Config
from src.data.splits import Fold, rows_for_clips
from src.features.source import FeatureSource
from src.models.base import Batch, FoldContext, WindowClassifier
from src.provenance import write as write_provenance
from src.results import checkpoint_path, model_directory
from src.train.folds import FoldPlan

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], WindowClassifier]


@dataclass
class CrossValidationResult:
    """Everything one model produced across every fold."""

    model_name: str
    config_name: str
    source_name: str
    clip_metrics: pd.DataFrame
    window_metrics: pd.DataFrame
    clip_predictions: pd.DataFrame
    confusion: pd.DataFrame
    extras: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def summary(self) -> pd.DataFrame:
        return scoring.summarise_folds(self.clip_metrics)

    def headline(self) -> str:
        return scoring.format_headline(self.model_name, self.summary)


def _batch(source: FeatureSource, rows: np.ndarray, labels: np.ndarray) -> Batch:
    return Batch(features=source.matrix(rows), labels=labels[rows])


@dataclass(frozen=True)
class FoldOutcome:
    """Everything one fold of one repeat produced."""

    clip_metrics: dict
    window_metrics: dict
    predictions: pd.DataFrame
    extras: dict[str, pd.DataFrame]


def _run_fold(
    cfg: Config,
    source: FeatureSource,
    fold: Fold,
    repeat: int,
    build_model: ModelFactory,
    model_name: str,
    class_names: list[str],
    labels: np.ndarray,
) -> FoldOutcome:
    """Fit on one fold and score the recordings it was never shown.

    Every row it returns carries both the repeat and the fold, because a comparison
    is paired on the pair: repeat three fold two of a model belongs with repeat three
    fold two of its control, since those are the same split.
    """
    index = source.index
    train_rows = rows_for_clips(index, fold.train_clips)
    validation_rows = rows_for_clips(index, fold.validation_clips)
    test_rows = rows_for_clips(index, fold.test_clips)
    if len(train_rows) == 0 or len(test_rows) == 0:
        raise RuntimeError(f"fold {fold.index} has an empty train or test partition")

    model = build_model()
    model.fit(
        _batch(source, train_rows, labels),
        _batch(source, validation_rows, labels),
        len(class_names),
    )

    window_probabilities = model.predict_proba(source.matrix(test_rows))
    clips, scores = scoring.evaluate_clips(index, test_rows, window_probabilities, class_names)

    context = FoldContext(
        fold_index=fold.index,
        feature_names=source.feature_names(),
        checkpoint=checkpoint_path(cfg, model_name, fold.index, repeat),
    )
    stamp = {"repeat": repeat, "fold": fold.index}

    logger.info(
        "  repeat %d fold %d: clip macro-F1 %.3f  acc %.3f  (%d test clips, %d windows)",
        repeat,
        fold.index,
        scores["macro_f1"],
        scores["accuracy"],
        len(clips),
        len(test_rows),
    )
    return FoldOutcome(
        clip_metrics={**stamp, **scores},
        window_metrics={
            **stamp,
            **scoring.score(labels[test_rows], window_probabilities, class_names),
        },
        predictions=clips.assign(**stamp),
        extras={key: table.assign(**stamp) for key, table in model.artifacts(context).items()},
    )


def run_cross_validation(
    cfg: Config,
    source: FeatureSource,
    plan: FoldPlan | list[Fold],
    build_model: ModelFactory,
    model_name: str,
    class_names: list[str] | None = None,
) -> CrossValidationResult:
    """Fit one model on every fold of every repeat, scoring it on unseen recordings.

    ``class_names`` defaults to the species under study. A task with a different
    label space, such as whether a clip contains a coda, passes its own; the runner
    otherwise has no way to know what the labels mean.

    ``plan`` may be a bare list of folds, which is one split, or a ``FoldPlan``
    carrying several. Repeats exist because five folds cannot separate differences
    of the size this project reports.
    """
    if isinstance(plan, list):
        plan = FoldPlan.single(plan)

    index = source.index
    labels = index["label"].to_numpy()
    class_names = list(class_names or cfg.dataset.species)

    clip_rows: list[dict] = []
    window_rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    extras: dict[str, list[pd.DataFrame]] = {}

    for repeat, folds in plan:
        for fold in folds:
            outcome = _run_fold(
                cfg, source, fold, repeat, build_model, model_name, class_names, labels
            )
            window_rows.append(outcome.window_metrics)
            clip_rows.append(outcome.clip_metrics)
            predictions.append(outcome.predictions)
            for key, table in outcome.extras.items():
                extras.setdefault(key, []).append(table)

    all_predictions = pd.concat(predictions, ignore_index=True)
    return CrossValidationResult(
        model_name=model_name,
        config_name=cfg.name,
        source_name=source.name,
        clip_metrics=pd.DataFrame(clip_rows),
        window_metrics=pd.DataFrame(window_rows),
        clip_predictions=all_predictions,
        confusion=scoring.confusion(
            all_predictions["label"].to_numpy(),
            all_predictions["prediction"].to_numpy(),
            class_names,
        ),
        extras={key: pd.concat(tables, ignore_index=True) for key, tables in extras.items()},
    )


def save_result(
    cfg: Config, result: CrossValidationResult, extra: dict[str, object] | None = None
) -> Path:
    """Write every artifact the report and the notebooks read.

    ``extra`` records anything about the run that the config alone does not say. A
    call type task that dropped its long clips has to write that down, or the
    directory looks like a run over every clip of that type and nothing contradicts
    it.
    """
    directory = model_directory(cfg, result.model_name)
    directory.mkdir(parents=True, exist_ok=True)

    result.clip_metrics.to_csv(directory / "fold_metrics_clip.csv", index=False)
    result.window_metrics.to_csv(directory / "fold_metrics_window.csv", index=False)
    result.summary.to_csv(directory / "summary.csv", index=False)
    result.confusion.to_csv(directory / "confusion.csv")
    result.clip_predictions.to_parquet(directory / "clip_predictions.parquet", index=False)
    for key, table in result.extras.items():
        table.to_csv(directory / f"{key}.csv", index=False)

    write_provenance(
        cfg,
        directory,
        extra={
            "model": result.model_name,
            "feature_source": result.source_name,
            "splits": len(result.clip_metrics),
            **(extra or {}),
        },
    )
    return directory
