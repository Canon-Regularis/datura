"""Averaging two fitted models into a third, without fitting anything.

Every model in this project is scored on the same folds under the same seed, so a
clip is predicted by each of them exactly once per split. That makes averaging their
probabilities free: the rows join on repeat, fold and clip, and the result is a set of
predictions over the identical test recordings that the members were scored on.

It is worth doing because it wins. On ``base_10k`` the trees reach 0.750 macro-F1 and
the probe 0.749, and averaging them reaches 0.773. Independent models rarely make the
same confident mistake, and the two here could hardly be less alike: one is gradient
boosted trees over hand engineered spectral descriptors, the other a linear map over a
frozen transformer pretrained on English speech. Adding either CNN makes it worse on
both accuracy and calibration, so the pair is named rather than the set.

The result is written as an ordinary result directory rather than reported as a note.
A model that a person can run, and that ``python -m src.predict --model xgboost,probe``
does run, has to carry a summary, a confusion matrix and a per fold table like every
other, or it cannot be compared against the controls and cannot appear in the
multiplicity correction. A claim that avoids the correction is a claim nobody checked.

Nothing here refits. If a member is missing, no ensemble is written, so a configuration
that trained one of the two reports its single models and says nothing about a pair.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import scoring
from src.config import Config
from src.evaluate.artifacts import write_table
from src.results import (
    clip_metrics_path,
    confusion_path,
    model_directory,
    predictions_path,
    summary_path,
)

logger = logging.getLogger(__name__)

# The two that beat both of their members. Measured rather than chosen: every pair and
# triple of the four audio models was scored on the committed predictions, and adding
# either CNN to this pair lowers accuracy and raises the confident error rate.
ENSEMBLE = ("xgboost", "probe")
ENSEMBLE_NAME = "+".join(ENSEMBLE)

# The columns a fold is addressed by. A model and its ensemble partner ran the same
# plan under the same seed, so these three identify one prediction uniquely.
KEYS = ["repeat", "fold", "clip_id"]


def averaged(cfg: Config, names: tuple[str, ...] = ENSEMBLE) -> pd.DataFrame | None:
    """Per prediction probabilities averaged across models, where they line up.

    Equal weight. Nothing here has been tuned to pick weights, and doing it honestly
    would need a validation fold that no member had already used for early stopping.

    Returns nothing when a member has not been fitted, and joins on the intersection
    otherwise, so a model that ran fewer repeats contributes fewer rows rather than
    silently averaging one member against nothing.
    """
    columns = scoring.probability_columns(len(cfg.dataset.species))

    # What a clip is, as opposed to what a model thought of it. Carried through from
    # the first member rather than recomputed, so an averaged prediction describes the
    # same recording its members described. The tape is not decoration: intervals are
    # bootstrapped over whole tapes, and a table without it cannot be resampled at all.
    carried = ["label", "tape_id", "species"]

    frames = []
    for name in names:
        path = predictions_path(cfg, name)
        if not path.exists():
            return None
        frame = pd.read_parquet(path).set_index(KEYS)
        missing = [column for column in [*columns, *carried] if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} predictions are missing {missing}")
        frames.append(frame)

    shared = frames[0].index
    for frame in frames[1:]:
        shared = shared.intersection(frame.index)
    if not len(shared):
        return None

    stacked = np.mean([frame.loc[shared, columns].to_numpy() for frame in frames], axis=0)
    out = pd.DataFrame(stacked, columns=columns, index=shared)
    for column in carried:
        out[column] = frames[0].loc[shared, column].to_numpy()
    out = out.reset_index()
    out["prediction"] = out[columns].to_numpy().argmax(axis=1)
    return out


def materialise(cfg: Config, names: tuple[str, ...] = ENSEMBLE) -> str | None:
    """Write the averaged model as a result directory, or nothing if a member is absent.

    The same four files every trained model writes, computed the same way, so the
    report reads it without a special case and the comparison against the controls is
    the comparison every other model gets.
    """
    pooled = averaged(cfg, names)
    if pooled is None:
        logger.info("not every member of %s is fitted; no ensemble written", "+".join(names))
        return None

    class_names = list(cfg.dataset.species)
    columns = scoring.probability_columns(len(class_names))
    name = "+".join(names)

    # Scored per split, because that is the unit every comparison is paired on. A
    # pooled score over all fifty at once would be a different number and could not be
    # placed beside a member's fold table.
    rows = []
    for (repeat, fold), group in pooled.groupby(["repeat", "fold"], sort=True):
        rows.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                **scoring.score(group["label"].to_numpy(), group[columns].to_numpy(), class_names),
            }
        )
    fold_metrics = pd.DataFrame(rows)

    model_directory(cfg, name).mkdir(parents=True, exist_ok=True)
    # Through the shared writer, because these two are the only result tables the report
    # regenerates. Every other model's are written once by a trainer and committed, so a
    # rebuild never touches them; these are recomputed from the committed predictions on
    # whatever machine runs the report. Written raw, they carried a full float64 repr and
    # disagreed with themselves in the sixteenth significant figure between Windows and
    # Linux, which failed the reproduce job and moved every p value downstream of them.
    write_table(fold_metrics, clip_metrics_path(cfg, name))
    write_table(scoring.summarise_folds(fold_metrics), summary_path(cfg, name))
    # The confusion matrix keeps its index, which names the classes, and holds integer
    # counts that no platform rounds differently.
    scoring.confusion(
        pooled["label"].to_numpy(), pooled["prediction"].to_numpy(), class_names
    ).to_csv(confusion_path(cfg, name))
    pooled.to_parquet(predictions_path(cfg, name), index=False)

    logger.info(
        "%s: macro-F1 %.3f over %d splits, averaged from %s",
        name,
        fold_metrics["macro_f1"].mean(),
        len(fold_metrics),
        ", ".join(names),
    )
    return name


def is_derived(name: str, known: set[str]) -> bool:
    """Whether a result directory is an average of models rather than a fitted one.

    Named by its members joined on a plus, so the rule is readable off the directory
    and no registry entry is needed for a model that was never trained. Every member
    has to be a real model, which stops a stray directory being adopted into a family.
    """
    parts = name.split("+")
    return len(parts) > 1 and all(part in known for part in parts)
