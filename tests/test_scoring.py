from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src import scoring
from src.evaluate.occlusion import band_edges
from src.scoring import aggregate_to_clips, confusion, score, summarise_folds
from tests.helpers import SPECIES as CLASSES
from tests.helpers import window_rows as window_index


def test_aggregation_averages_windows_within_a_clip():
    index = window_index({"1000001": (0, 3)})
    probabilities = np.array([[0.9, 0.1, 0.0], [0.3, 0.7, 0.0], [0.6, 0.2, 0.2]])

    clips = aggregate_to_clips(index, np.arange(3), probabilities)

    assert len(clips) == 1
    assert clips.loc[0, "p0"] == pytest.approx(0.6)
    assert clips.loc[0, "p1"] == pytest.approx(1.0 / 3)
    assert clips.loc[0, "prediction"] == 0


def test_a_clip_counts_once_however_many_windows_it_has():
    """A long recording must not outvote a short one just by being longer."""
    index = window_index({"1000001": (0, 16), "2000001": (1, 1)})
    probabilities = np.zeros((17, 3))
    probabilities[:16, 0] = 1.0
    probabilities[16, 1] = 1.0

    clips = aggregate_to_clips(index, np.arange(17), probabilities)

    assert len(clips) == 2
    assert set(clips["clip_id"]) == {"1000001", "2000001"}


def test_aggregation_rejects_a_length_mismatch():
    index = window_index({"1000001": (0, 3)})
    with pytest.raises(ValueError, match="probability vectors"):
        aggregate_to_clips(index, np.arange(3), np.zeros((2, 3)))


def test_aggregation_only_uses_the_requested_rows():
    index = window_index({"1000001": (0, 2), "2000001": (1, 2)})
    clips = aggregate_to_clips(index, np.array([2, 3]), np.array([[0.1, 0.9, 0.0]] * 2))

    assert list(clips["clip_id"]) == ["2000001"]


def test_score_reports_every_class_even_when_absent():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0]])

    result = score(labels, probabilities, CLASSES)

    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["support_KillerWhale"] == 0.0
    assert np.isfinite(result["roc_auc_ovr_macro"])


def test_score_survives_a_single_class_fold():
    result = score(np.zeros(4, dtype=int), np.tile([1.0, 0, 0], (4, 1)), CLASSES)
    assert np.isnan(result["roc_auc_ovr_macro"])


def test_confusion_keeps_class_order():
    matrix = confusion(np.array([0, 1, 2]), np.array([0, 1, 1]), CLASSES)

    assert tuple(matrix.index) == CLASSES
    assert matrix.loc["KillerWhale", "SpermWhale"] == 1


def test_summary_carries_a_spread():
    folds = pd.DataFrame({"fold": [0, 1, 2], "macro_f1": [0.6, 0.8, 0.7]})
    summary = summarise_folds(folds).set_index("metric")

    assert "fold" not in summary.index
    assert summary.loc["macro_f1", "mean"] == pytest.approx(0.7)
    assert summary.loc["macro_f1", "std"] == pytest.approx(0.1)


def test_band_edges_cover_every_mel_bin_once():
    edges = band_edges(64, 8)

    assert edges[0][0] == 0
    assert edges[-1][1] == 64
    assert all(high == nxt for (_, high), (nxt, _) in pairwise(edges))
    assert sum(high - low for low, high in edges) == 64


def test_band_edges_handle_more_groups_than_bins():
    edges = band_edges(4, 8)
    assert sum(high - low for low, high in edges) == 4


def test_the_count_metrics_match_sklearn_exactly():
    """The bootstrap scores from confusion counts rather than through sklearn.

    It runs thousands of times, and the ranking metrics it never asks for cost more
    than everything else together. That shortcut is only safe while it agrees with
    the library to the last digit, including where a class is absent from a fold.
    """
    rng = np.random.default_rng(0)
    for n_classes in (2, 3, 5):
        all_labels = list(range(n_classes))
        for _ in range(50):
            size = int(rng.integers(5, 200))
            labels = rng.integers(0, n_classes, size)
            predictions = rng.integers(0, n_classes, size)

            counted = scoring.from_counts(labels, predictions, n_classes)
            assert counted["accuracy"] == pytest.approx(accuracy_score(labels, predictions))
            assert counted["macro_f1"] == pytest.approx(
                f1_score(labels, predictions, labels=all_labels, average="macro", zero_division=0)
            )
            assert counted["weighted_f1"] == pytest.approx(
                f1_score(
                    labels, predictions, labels=all_labels, average="weighted", zero_division=0
                )
            )


def test_an_empty_set_of_predictions_scores_zero_rather_than_dividing_by_nothing():
    counted = scoring.from_counts(np.array([], dtype=int), np.array([], dtype=int), 3)
    assert counted == {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}


def test_window_predictions_keep_the_position_inside_the_clip():
    """The only time coordinate this pipeline produces, and nothing read it before.

    Scoring is done on clips because windows of one clip are not independent, but
    the per window scores were computed and discarded. Keeping them, with the
    window's position in its clip, is what any later work on where in a recording a
    call happens would start from.
    """
    from src.train.crossval import _window_frame

    index = pd.DataFrame(
        {
            "clip_id": ["a", "a", "b"],
            "tape_id": ["t", "t", "t"],
            "species": ["x", "x", "y"],
            "label": [0, 0, 1],
            "window_index": [0, 1, 0],
        }
    )
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])

    frame = _window_frame(index, np.array([0, 1, 2]), probabilities, {"repeat": 0, "fold": 3})

    assert list(frame["window_index"]) == [0, 1, 0]
    assert list(frame["clip_id"]) == ["a", "a", "b"]
    assert frame["p1"].tolist() == pytest.approx([0.1, 0.4, 0.8])
    assert set(frame["fold"]) == {3}


def test_a_summary_says_how_many_folds_each_metric_rests_on():
    """A ranking metric is undefined on a fold holding one class.

    pandas averages around the gap without saying so, which put two rows of the same
    table on different numbers of folds and made them look alike.
    """
    folds = pd.DataFrame(
        {
            "fold": [0, 1, 2],
            "macro_f1": [0.6, 0.7, 0.8],
            "roc_auc_ovr_macro": [0.9, float("nan"), 0.7],
        }
    )
    summary = summarise_folds(folds).set_index("metric")

    assert summary.loc["macro_f1", "folds"] == 3
    assert summary.loc["roc_auc_ovr_macro", "folds"] == 2
    assert summary.loc["roc_auc_ovr_macro", "mean"] == pytest.approx(0.8)
