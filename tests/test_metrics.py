from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from src.evaluate.metrics import aggregate_to_clips, confusion, score, summarise_folds
from src.evaluate.occlusion import band_edges

CLASSES = ["HumpbackWhale", "SpermWhale", "KillerWhale"]


def window_index(windows_per_clip: dict[str, tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for clip_id, (label, count) in windows_per_clip.items():
        for position in range(count):
            rows.append(
                {
                    "clip_id": clip_id,
                    "tape_id": clip_id[:5],
                    "species": CLASSES[label],
                    "label": label,
                    "window_index": position,
                }
            )
    return pd.DataFrame(rows)


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

    assert list(matrix.index) == CLASSES
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
