"""The shared cross-validation runner.

One loop fits every model in the project. It never learns what kind of model it is
holding, so the acoustic baseline, the CNN and the metadata control are evaluated
under identical folds, identical aggregation and identical metrics. The harness is the
same for all of them, so any gap between their reported numbers came from the features.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.splits import Fold, rows_for_clips
from src.evaluate import metrics
from src.features.source import FeatureSource
from src.models.base import Batch, WindowClassifier
from src.provenance import write as write_provenance

ModelFactory = Callable[[], WindowClassifier]
FoldHook = Callable[[int, WindowClassifier], dict[str, pd.DataFrame]]


@dataclass
class CrossValidationResult:
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
        return metrics.summarise_folds(self.clip_metrics)

    def headline(self) -> str:
        return metrics.format_headline(self.model_name, self.summary)


def _batch(source: FeatureSource, rows: np.ndarray, labels: np.ndarray) -> Batch:
    return Batch(features=source.matrix(rows), labels=labels[rows])


def run_cross_validation(
    cfg: Config,
    source: FeatureSource,
    folds: list[Fold],
    build_model: ModelFactory,
    model_name: str,
    fold_hook: FoldHook | None = None,
    verbose: bool = True,
) -> CrossValidationResult:
    index = source.index
    labels = index["label"].to_numpy()
    class_names = list(cfg.dataset.species)
    n_classes = len(class_names)

    clip_rows: list[dict] = []
    window_rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    extras: dict[str, list[pd.DataFrame]] = {}

    for fold in folds:
        train_rows = rows_for_clips(index, fold.train_clips)
        validation_rows = rows_for_clips(index, fold.validation_clips)
        test_rows = rows_for_clips(index, fold.test_clips)
        if len(train_rows) == 0 or len(test_rows) == 0:
            raise RuntimeError(f"fold {fold.index} has an empty train or test partition")

        model = build_model()
        model.fit(
            _batch(source, train_rows, labels),
            _batch(source, validation_rows, labels),
            n_classes,
        )

        window_probabilities = model.predict_proba(source.matrix(test_rows))
        window_rows.append(
            {
                "fold": fold.index,
                **metrics.score(labels[test_rows], window_probabilities, class_names),
            }
        )

        clips = metrics.aggregate_to_clips(index, test_rows, window_probabilities)
        clip_probabilities = clips[[f"p{i}" for i in range(n_classes)]].to_numpy()
        clip_rows.append(
            {
                "fold": fold.index,
                **metrics.score(clips["label"].to_numpy(), clip_probabilities, class_names),
            }
        )
        predictions.append(clips.assign(fold=fold.index))

        if fold_hook is not None:
            for key, table in fold_hook(fold.index, model).items():
                extras.setdefault(key, []).append(table.assign(fold=fold.index))

        if verbose:
            print(
                f"  fold {fold.index}: "
                f"clip macro-F1 {clip_rows[-1]['macro_f1']:.3f}  "
                f"acc {clip_rows[-1]['accuracy']:.3f}  "
                f"({len(clips)} test clips, {len(test_rows)} windows)"
            )

    all_predictions = pd.concat(predictions, ignore_index=True)
    return CrossValidationResult(
        model_name=model_name,
        config_name=cfg.name,
        source_name=source.name,
        clip_metrics=pd.DataFrame(clip_rows),
        window_metrics=pd.DataFrame(window_rows),
        clip_predictions=all_predictions,
        confusion=metrics.confusion(
            all_predictions["label"].to_numpy(),
            all_predictions["prediction"].to_numpy(),
            class_names,
        ),
        extras={key: pd.concat(tables, ignore_index=True) for key, tables in extras.items()},
    )


def result_directory(cfg: Config, model_name: str) -> Path:
    return cfg.paths.reports / cfg.name / model_name


def save_result(cfg: Config, result: CrossValidationResult) -> Path:
    """Write every artifact the report and the notebooks read."""
    directory = result_directory(cfg, result.model_name)
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
        extra={"model": result.model_name, "feature_source": result.source_name},
    )
    return directory
