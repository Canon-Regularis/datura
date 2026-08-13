"""What the audio identifies, other than the species it was asked about.

The place held out result says a model trained on other places is worth nothing at a new
one, and three explanations fit it: the recording channel changed, the animals changed
because dialects are regional, or the class mix changed. The report cannot separate
them, because every question it asks is about the species.

These questions are about the recording instead. Each is posed inside a single species,
so species is not the answer to any of them, and each is read against the macro-F1 that
drawing from the class shares would reach, the same floor the report quotes.

Which tape did this clip come from. Clips of one tape are cuts of one continuous
recording, so content similarity contributes and this is an upper bound on the channel
signature rather than a measurement of it. It is worth having for the comparison: a
fifty way tape question that is easier than the three way species question says per
recording identity is the stronger signal in these descriptors.

Was this unseen tape recorded at the anchor place. Whole tapes are held out, so a model
has to carry a location signature to a recording it has never heard rather than
recognise one it has. It is binary because the places are far too lopsided for anything
else, and the anchor is whichever place holds most of the species' tapes.

What rate was this unseen tape recorded at. Every clip is resampled to one rate so all
species share a band, and this asks whether that erased the recorder. It is the question
that names the mechanism, because ``native_sample_rate`` is one of the four fields the
metadata control reads.

Usage:
    python -m src.evaluate.diagnostics [--config configs/base.yaml]
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src import cli, scoring
from src.config import Config
from src.data.manifest import load_manifest
from src.data.splits import rows_for_clips
from src.evaluate.artifacts import write_table
from src.features import registry as features
from src.features.source import CentredSource, FeatureSource
from src.models.base import Batch
from src.models.gbdt import GradientBoostedTrees
from src.results import diagnostics_path

logger = logging.getLogger(__name__)

# Deliberately lighter than the published model. These questions ask whether a signal is
# present at all rather than what the best achievable score is, and a fifty class
# question at the committed depth takes hours to answer something a shallow model
# answers. Pinned here so the numbers are reproducible rather than incidental.
SETTINGS = {
    "n_estimators": 60,
    "max_depth": 4,
    "learning_rate": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": 6,
    "random_state": 1234,
}
FOLDS = 4
SEED = 1234

# A tape needs enough clips to sit on both sides of every fold, or a training fold is
# missing one of the classes and the estimator refuses the label set.
MINIMUM_CLIPS = 2 * FOLDS

# The rates a resampler would treat differently, rather than the exact figure, because
# the recorded rates are near continuous and a class per rate would leave one tape in
# most of them.
RATE_EDGES = [0, 15_000, 25_000, 50_000, 100_000, np.inf]

COLUMNS = (
    "question",
    "species",
    "treatment",
    "macro_f1",
    "macro_f1_std",
    "guessing",
    "n_classes",
    "n_clips",
    "held_out",
)


def guessing_floor(labels: np.ndarray, draws: int = 30) -> float:
    """Macro-F1 of drawing from the class shares, which is what a score is read against.

    Not the accuracy of that draw, which is ``sum(p**2)`` and a different number. Reading
    a macro-F1 against it says a model is below chance when it is above, and doing so is
    a mistake this file exists partly to stop repeating.
    """
    n_classes = int(labels.max()) + 1
    shares = np.bincount(labels, minlength=n_classes) / len(labels)
    rng = np.random.default_rng(SEED)
    scores = [
        scoring.from_counts(labels, rng.choice(n_classes, len(labels), p=shares), n_classes)[
            "macro_f1"
        ]
        for _ in range(draws)
    ]
    return float(np.mean(scores))


def _splits(clips: pd.DataFrame, *, hold_out_tapes: bool):
    if hold_out_tapes:
        splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        return splitter.split(clips, clips["label"], groups=clips["tape_id"])
    # The tape cannot be held out when the tape is the answer, so clips are.
    return StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
        clips, clips["label"]
    )


def cross_validate(source: FeatureSource, index: pd.DataFrame, *, hold_out_tapes: bool):
    """One question, scored on held out clips or held out tapes."""
    labels = index["label"].to_numpy()
    n_classes = int(labels.max()) + 1
    names = [str(label) for label in range(n_classes)]
    clips = (
        index.groupby("clip_id")
        .agg(label=("label", "first"), tape_id=("tape_id", "first"))
        .reset_index()
    )

    macros = []
    for train_index, test_index in _splits(clips, hold_out_tapes=hold_out_tapes):
        train = rows_for_clips(index, list(clips.iloc[train_index]["clip_id"]))
        test = rows_for_clips(index, list(clips.iloc[test_index]["clip_id"]))
        model = GradientBoostedTrees(SETTINGS)
        # The held out rows double as the early stopping set. That would be cheating for
        # a published score and is fine here, where the question is whether a signal
        # exists at all and a leak works against the null rather than for it.
        model.fit(
            Batch(source.matrix(train), labels[train]),
            Batch(source.matrix(test), labels[test]),
            n_classes,
        )
        scored = scoring.aggregate_to_clips(index, test, model.predict_proba(source.matrix(test)))
        truth = index.iloc[test].groupby("clip_id")["label"].first().reindex(scored["clip_id"])
        macros.append(
            scoring.score(
                truth.to_numpy(), np.eye(n_classes)[scored["prediction"].to_numpy()], names
            )["macro_f1"]
        )

    return {
        "macro_f1": float(np.mean(macros)),
        "macro_f1_std": float(np.std(macros, ddof=1)),
        "guessing": guessing_floor(labels),
        "n_classes": n_classes,
        "n_clips": len(clips),
    }


class _Subset(FeatureSource):
    """Some rows of another source, renumbered so a fold index addresses them."""

    def __init__(self, base: FeatureSource, rows: np.ndarray, index: pd.DataFrame):
        self._base = base
        self._rows = np.asarray(rows, dtype=np.int64)
        self._index = index.reset_index(drop=True)

    @property
    def name(self) -> str:
        return f"{self._base.name}_subset"

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def matrix(self, rows):
        return self._base.matrix(self._rows[np.asarray(rows, dtype=np.int64)])

    def feature_names(self):
        return self._base.feature_names()


def _labelled(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    return frame.assign(label=pd.factorize(labels)[0])


def questions(cfg: Config, source: FeatureSource, treatment: str) -> list[dict]:
    """Every question, answered on one treatment of the features."""
    manifest = load_manifest(cfg, kept_only=True).set_index("clip_id")
    index = source.index.reset_index(drop=True)
    index = index.assign(
        place=index["clip_id"].map(manifest["place"]).fillna(""),
        rate=index["clip_id"].map(manifest["native_sample_rate"]),
    )

    logger.info("%s features: the species question first, for scale", treatment)
    rows: list[dict] = [
        {
            "question": "species",
            "species": "all",
            "treatment": treatment,
            "held_out": "tapes",
            **cross_validate(source, index, hold_out_tapes=True),
        }
    ]

    logger.info("  species from audio: %.3f", rows[0]["macro_f1"])

    for species in sorted(cfg.dataset.species):
        positions = np.flatnonzero(index["species"].to_numpy() == species)
        if not len(positions):
            continue
        logger.info("  %s, asking what else the audio knows", species)
        part = index.iloc[positions].reset_index(drop=True)

        per_tape = part.groupby("tape_id")["clip_id"].nunique()
        thick = set(per_tape[per_tape >= MINIMUM_CLIPS].index)
        if len(thick) >= 2:
            keep = np.flatnonzero(part["tape_id"].isin(thick).to_numpy())
            subset = part.iloc[keep].reset_index(drop=True)
            rows.append(
                {
                    "question": "tape",
                    "species": species,
                    "treatment": treatment,
                    "held_out": "clips",
                    **cross_validate(
                        _Subset(source, positions[keep], _labelled(subset, subset["tape_id"])),
                        _labelled(subset, subset["tape_id"]),
                        hold_out_tapes=False,
                    ),
                }
            )

        placed = np.flatnonzero(part["place"].to_numpy() != "")
        located = part.iloc[placed].reset_index(drop=True)
        if len(located):
            anchor = located.groupby("place")["tape_id"].nunique().idxmax()
            labelled = located.assign(label=(located["place"] == anchor).astype(int))
            if labelled.groupby("label")["tape_id"].nunique().min() >= FOLDS:
                rows.append(
                    {
                        "question": f"place is {anchor}",
                        "species": species,
                        "treatment": treatment,
                        "held_out": "tapes",
                        **cross_validate(
                            _Subset(source, positions[placed], labelled),
                            labelled,
                            hold_out_tapes=True,
                        ),
                    }
                )

        band = pd.cut(part["rate"], RATE_EDGES, labels=False, right=False)
        counts = part.groupby(band)["tape_id"].nunique()
        wanted = set(counts[counts >= FOLDS].index)
        if len(wanted) >= 2:
            keep = np.flatnonzero(band.isin(wanted).to_numpy())
            subset = part.iloc[keep].reset_index(drop=True)
            rows.append(
                {
                    "question": "native rate band",
                    "species": species,
                    "treatment": treatment,
                    "held_out": "tapes",
                    **cross_validate(
                        _Subset(source, positions[keep], _labelled(subset, band.iloc[keep])),
                        _labelled(subset, band.iloc[keep]),
                        hold_out_tapes=True,
                    ),
                }
            )

    for row in rows[1:]:
        logger.info(
            "    %-18s %.3f against %.3f", row["question"], row["macro_f1"], row["guessing"]
        )
    return rows


def build(cfg: Config) -> pd.DataFrame:
    """Answer every question on each treatment of the descriptors."""
    raw = features.load_source(features.ACOUSTIC, cfg)
    rows = questions(cfg, raw, "raw")
    rows += questions(cfg, CentredSource(raw, name="centred"), "centred")
    # The stronger normalisation, measured rather than assumed to be an improvement. It
    # is not one, which is why ``acoustic_centred`` does not use it.
    rows += questions(cfg, CentredSource(raw, name="whitened", scale=True), "whitened")

    table = pd.DataFrame(rows, columns=list(COLUMNS))
    destination = diagnostics_path(cfg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The same writer every other committed table goes through. Rounding here to four
    # places and quoting three in the README double rounds 0.7575 into disagreeing with
    # itself, which is the failure this writer exists to stop.
    write_table(table, destination)
    logger.info("\n%s", table.round(3).to_string(index=False))
    logger.info("written to %s", destination)
    return table


def main(argv: list[str] | None = None) -> int:
    build(cli.prepare(cli.parser_for(__doc__).parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
