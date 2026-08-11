"""What a model is worth when it is allowed to decline.

The curve is the only thing in this repository that describes a model the way somebody
using it would experience one. Everything else forces an answer for every clip, which
is right for comparing two representations and wrong for describing a tool.

It also has to be trustworthy in a specific way. A prediction command reads a
threshold off it and prints a confidence claim, so a curve that overstated accuracy
would put a number beside an answer that the data does not support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import scoring
from src.config import load_config
from src.evaluate import coverage
from src.results import config_directory, predictions_path

CONFIG = "configs/base.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def curve(cfg):
    path = config_directory(cfg) / "coverage.csv"
    if not path.exists():
        pytest.skip("coverage.csv absent; run python -m src.evaluate.report first")
    return pd.read_csv(path)


def test_full_coverage_reproduces_the_score_reported_everywhere_else(cfg, curve):
    """If it did not, the curve would be describing a different set of predictions."""
    for name in ("xgboost", "cnn_small"):
        if not predictions_path(cfg, name).exists():
            continue
        predictions = pd.read_parquet(predictions_path(cfg, name))
        columns = scoring.probability_columns(len(cfg.dataset.species))
        pooled = scoring.from_counts(
            predictions["label"].to_numpy(),
            predictions[columns].to_numpy().argmax(axis=1),
            len(cfg.dataset.species),
        )
        row = curve[(curve["model"] == name) & (curve["coverage"] == 1.0)].iloc[0]
        assert row["accuracy"] == pytest.approx(pooled["accuracy"], abs=5e-4)


def test_declining_the_least_confident_raises_accuracy(curve):
    """The property that makes abstention worth having at all."""
    for name, rows in curve.groupby("model"):
        ordered = rows.sort_values("coverage", ascending=False)
        full = ordered.iloc[0]["accuracy"]
        tightest = ordered.iloc[-1]["accuracy"]
        assert tightest > full, f"{name} gains nothing by declining, so ranking is useless"


def test_the_threshold_rises_as_coverage_falls(curve):
    for _, rows in curve.groupby("model"):
        ordered = rows.sort_values("coverage", ascending=False)
        assert ordered["threshold"].is_monotonic_increasing


def test_coverage_matches_the_share_of_predictions_kept(curve):
    for _, rows in curve.groupby("model"):
        full = rows[rows["coverage"] == 1.0].iloc[0]["predictions"]
        for _, row in rows.iterrows():
            assert row["predictions"] / full == pytest.approx(row["coverage"], abs=0.02)


def test_the_ensemble_beats_both_of_its_members(cfg, curve):
    """The reason it is in the table rather than a note in a docstring.

    Averaging the trees and the probe is better than either alone at full coverage,
    and it is better because independent models rarely make the same confident
    mistake. Adding either network makes it worse, which is why the pair is named.
    """
    if coverage.ENSEMBLE_NAME not in set(curve["model"]):
        pytest.skip("the ensemble members have not both been fitted")

    at_full = curve[curve["coverage"] == 1.0].set_index("model")
    combined = at_full.loc[coverage.ENSEMBLE_NAME]
    for member in coverage.ENSEMBLE:
        assert combined["accuracy"] > at_full.loc[member, "accuracy"]
        assert combined["macro_f1"] > at_full.loc[member, "macro_f1"]


def test_the_ensemble_makes_fewer_confident_mistakes_than_its_members(cfg):
    """The failure mode a threshold cannot catch, which is why this is measured.

    Abstention filters uncertainty, not error. A model that is wrong while sure stays
    wrong while sure however the cut off moves, so the only lever is a model that is
    confidently wrong less often.
    """
    pooled = coverage.averaged(cfg)
    if pooled is None:
        pytest.skip("the ensemble members have not both been fitted")

    columns = scoring.probability_columns(len(cfg.dataset.species))

    def confidently_wrong(frame: pd.DataFrame) -> float:
        probabilities = frame[columns].to_numpy()
        wrong = probabilities.argmax(axis=1) != frame["label"].to_numpy()
        return float((wrong & (probabilities.max(axis=1) > 0.9)).mean())

    together = confidently_wrong(pooled)
    for member in coverage.ENSEMBLE:
        alone = confidently_wrong(pd.read_parquet(predictions_path(cfg, member)))
        assert together <= alone + 0.005, f"averaging is worse than {member} alone"


def test_a_band_describes_the_confidence_it_was_asked_about(curve):
    rows = curve[curve["model"] == "xgboost"]
    matched = coverage.band(rows, 0.99)

    assert matched is not None
    assert matched["threshold"] <= 0.99
    assert coverage.band(rows, 0.0) is None, "nothing should match below every threshold"


def test_the_controls_are_not_given_an_operating_curve(curve):
    """A curve for the logbook would describe how confidently it reads paperwork."""
    assert not {"logbook", "metadata"} & set(curve["model"])


def test_averaging_needs_every_member_present(cfg):
    assert coverage.averaged(cfg, ("xgboost", "not_a_model")) is None


def test_averaging_lines_predictions_up_by_split_and_clip(cfg):
    """Every model saw the same folds, so a clip joins to itself rather than to another."""
    pooled = coverage.averaged(cfg)
    if pooled is None:
        pytest.skip("the ensemble members have not both been fitted")

    trees = pd.read_parquet(predictions_path(cfg, "xgboost"))
    keys = ["repeat", "fold", "clip_id"]
    labels = trees.set_index(keys)["label"]

    joined = pooled.set_index(keys)
    assert (joined["label"] == labels.loc[joined.index]).all(), "rows joined to the wrong clip"

    columns = scoring.probability_columns(len(cfg.dataset.species))
    assert np.allclose(pooled[columns].to_numpy().sum(axis=1), 1.0)
