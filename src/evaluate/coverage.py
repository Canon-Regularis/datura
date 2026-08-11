"""What a model is worth when it is allowed to decline.

Every score elsewhere in this project forces the model to answer for every clip. That
is the right way to compare two representations and the wrong way to describe a tool
somebody would use, because a classifier that says "I do not know" on the hard third
of its input is more useful than one that guesses.

This ranks the held out predictions by how confident the model was and reports what
accuracy survives at each level of coverage. The result is an operating curve rather
than a single number: at full coverage XGBoost is right about 83% of the time, and on
the 70% of clips it is most sure about, about 90%.

Nothing is refitted. The per clip probabilities were already committed, so this reads
what the cross validation wrote and asks a different question of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import scoring
from src.config import Config
from src.evaluate.ensemble import ENSEMBLE, ENSEMBLE_NAME, averaged
from src.results import predictions_path

__all__ = ["ENSEMBLE", "ENSEMBLE_NAME", "LEVELS", "averaged", "band", "for_model", "table"]

# Read top down. Full coverage is the number reported everywhere else, and each row
# below it is the model declining the clips it was least sure about.
LEVELS = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def for_model(cfg: Config, name: str, levels: tuple[float, ...] = LEVELS) -> pd.DataFrame:
    """One row per coverage level for one model.

    Confidence is the probability of the class the model chose. Ranking on it and
    cutting at a quantile is the simplest rule that could work, and it is the one an
    operator can actually implement: everything here is available at prediction time,
    with no label required.

    Predictions are pooled over every split. A repeated run holds one row per clip per
    repeat, so a threshold here is a statement about predictions rather than clips.
    """
    return _curve(cfg, name, pd.read_parquet(predictions_path(cfg, name)), levels)


def _curve(
    cfg: Config,
    name: str,
    predictions: pd.DataFrame,
    levels: tuple[float, ...] = LEVELS,
) -> pd.DataFrame:
    """The operating curve for one set of per clip probabilities."""
    class_names = list(cfg.dataset.species)
    columns = scoring.probability_columns(len(class_names))
    probabilities = predictions[columns].to_numpy()
    labels = predictions["label"].to_numpy()
    chosen = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    rows = []
    for level in levels:
        # The threshold that keeps this share of predictions, taken from the data
        # rather than from a grid, so every row holds exactly the coverage it claims.
        threshold = float(np.quantile(confidence, 1.0 - level))
        keep = confidence >= threshold
        if not keep.any():
            continue

        counts = scoring.from_counts(labels[keep], chosen[keep], len(class_names))
        rows.append(
            {
                "model": name,
                "coverage": round(float(keep.mean()), 4),
                "threshold": round(threshold, 4),
                "predictions": int(keep.sum()),
                "clips": int(predictions.loc[keep, "clip_id"].nunique()),
                "accuracy": counts["accuracy"],
                "macro_f1": counts["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def table(cfg: Config, model_names: list[str]) -> pd.DataFrame:
    """The operating curve for every model that hears the recording.

    The controls are left out on purpose. A curve for the logbook would describe how
    confidently it reads the paperwork, which is not a question anyone is asking of a
    tool that takes a wav file.

    The ensemble needs no special case here. ``src.evaluate.ensemble`` writes it as an
    ordinary result before the report reads any of them, so it arrives in
    ``model_names`` like anything else that hears the recording.
    """
    frames = [for_model(cfg, name) for name in model_names]

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def band(curve: pd.DataFrame, confidence: float) -> pd.Series | None:
    """The tightest coverage band a given confidence still falls inside.

    What a prediction command needs: turn one probability into a statement about how
    often the model is right when it is this sure, taken from held out data rather
    than invented.
    """
    inside = curve[curve["threshold"] <= confidence]
    return inside.iloc[-1] if len(inside) else None
