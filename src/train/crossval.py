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
import os
import time
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
from src.results import (
    calibrated_metrics_path,
    checkpoint_path,
    clip_metrics_path,
    confusion_path,
    model_directory,
    predictions_path,
    summary_path,
    validation_metrics_path,
    window_metrics_path,
    window_predictions_path,
)
from src.train.folds import FoldPlan

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], WindowClassifier]


@dataclass
class CrossValidationResult:
    """Everything one model produced across every fold."""

    model_name: str
    source_name: str
    clip_metrics: pd.DataFrame
    window_metrics: pd.DataFrame
    # Scored on rows held out of training for early stopping. Anything selected by
    # looking at a score is selected on these, never on the test rows.
    calibrated_metrics: pd.DataFrame
    validation_metrics: pd.DataFrame
    validation_predictions: pd.DataFrame
    clip_predictions: pd.DataFrame
    window_predictions: pd.DataFrame
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
    # The same test clips, decided with per class weights fitted on validation. Kept
    # beside the plain scores rather than replacing them, so the two are comparable and
    # nothing already published moves until the comparison says it should.
    calibrated_metrics: dict
    validation_metrics: dict
    validation_predictions: pd.DataFrame
    predictions: pd.DataFrame
    window_predictions: pd.DataFrame
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

    # The validation rows were held out of training for early stopping, and scoring them
    # costs one more forward pass. Anything chosen by looking at a number, a decision
    # threshold or a spectrogram setting, has to be chosen here rather than on the test
    # rows, or the figure that gets published is the one the choice was made on.
    validation_clips, validation_scores = scoring.evaluate_clips(
        index, validation_rows, model.predict_proba(source.matrix(validation_rows)), class_names
    )

    # One multiplier per class, chosen on the validation clips and applied to the test
    # clips. Fitted here rather than after the run because these are the only rows the
    # model was neither trained on nor is about to be judged on.
    weights = scoring.fit_decision_weights(
        validation_clips["label"].to_numpy(),
        validation_clips[scoring.probability_columns(len(class_names))].to_numpy(),
        class_names,
    )
    _, calibrated_scores = scoring.evaluate_clips(
        index, test_rows, window_probabilities, class_names, weights=weights
    )

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
        calibrated_metrics={
            **stamp,
            **calibrated_scores,
            **{f"weight_{name}": float(w) for name, w in zip(class_names, weights, strict=True)},
        },
        validation_metrics={**stamp, **validation_scores},
        validation_predictions=validation_clips.assign(**stamp),
        window_metrics={
            **stamp,
            **scoring.score(labels[test_rows], window_probabilities, class_names),
        },
        predictions=clips.assign(**stamp),
        window_predictions=_window_frame(index, test_rows, window_probabilities, stamp),
        extras={key: table.assign(**stamp) for key, table in model.artifacts(context).items()},
    )


def _window_frame(
    index: pd.DataFrame, rows: np.ndarray, probabilities: np.ndarray, stamp: dict
) -> pd.DataFrame:
    """One row per test window, before anything is averaged up to the clip.

    Scoring happens on clips, because windows of one clip are not independent. The
    per window scores were being computed and thrown away, and the position of a
    window inside its clip is a time coordinate nothing has ever read. Keeping both
    costs one small parquet and is what any later work on where in a recording a
    call happens would have to start from.

    A caveat that belongs with the file: ``max_windows_per_clip`` thins long clips
    to an even spread, so ``window_index`` is a position among the windows that were
    kept rather than a uniform grid over the recording.
    """
    columns = [name for name in ("clip_id", "tape_id", "window_index", "label") if name in index]
    frame = index.iloc[rows].loc[:, columns].reset_index(drop=True)
    scores = pd.DataFrame(
        probabilities, columns=scoring.probability_columns(probabilities.shape[1])
    )
    return pd.concat([frame, scores], axis=1).assign(**stamp)


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
    calibrated_rows: list[dict] = []
    validation_rows: list[dict] = []
    validation_predictions: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    window_predictions: list[pd.DataFrame] = []
    extras: dict[str, list[pd.DataFrame]] = {}

    cooldown = float(os.environ.get("DATURA_FOLD_COOLDOWN", "0"))
    if cooldown:
        logger.info("pausing %.0fs between folds, so the machine can shed heat", cooldown)

    for repeat, folds in plan:
        for fold in folds:
            if cooldown and (clip_rows or window_rows):
                time.sleep(cooldown)
            outcome = _run_fold(
                cfg, source, fold, repeat, build_model, model_name, class_names, labels
            )
            window_rows.append(outcome.window_metrics)
            clip_rows.append(outcome.clip_metrics)
            calibrated_rows.append(outcome.calibrated_metrics)
            validation_rows.append(outcome.validation_metrics)
            validation_predictions.append(outcome.validation_predictions)
            predictions.append(outcome.predictions)
            if repeat == 0:
                # One complete pass is enough. Every repeat partitions the same
                # clips, so ten of them would be ten copies of the same windows.
                window_predictions.append(outcome.window_predictions)
            for key, table in outcome.extras.items():
                extras.setdefault(key, []).append(table)

    all_predictions = pd.concat(predictions, ignore_index=True)
    return CrossValidationResult(
        model_name=model_name,
        source_name=source.name,
        clip_metrics=pd.DataFrame(clip_rows),
        window_metrics=pd.DataFrame(window_rows),
        calibrated_metrics=pd.DataFrame(calibrated_rows),
        validation_metrics=pd.DataFrame(validation_rows),
        validation_predictions=pd.concat(validation_predictions, ignore_index=True),
        clip_predictions=all_predictions,
        window_predictions=pd.concat(window_predictions, ignore_index=True),
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
    name = result.model_name
    directory = model_directory(cfg, name)
    directory.mkdir(parents=True, exist_ok=True)

    result.clip_metrics.to_csv(clip_metrics_path(cfg, name), index=False)
    result.validation_metrics.to_csv(validation_metrics_path(cfg, name), index=False)
    result.calibrated_metrics.to_csv(calibrated_metrics_path(cfg, name), index=False)
    result.window_metrics.to_csv(window_metrics_path(cfg, name), index=False)
    result.summary.to_csv(summary_path(cfg, name), index=False)
    result.confusion.to_csv(confusion_path(cfg, name))
    result.clip_predictions.to_parquet(predictions_path(cfg, name), index=False)
    result.window_predictions.to_parquet(window_predictions_path(cfg, name), index=False)
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
