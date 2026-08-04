"""What each control is allowed to see, and in what order.

Every published margin is a distance from one of these. A column quietly added,
reordered or recoded moves numbers that are already written down, so the shape of
each control is pinned here rather than left to whoever edits it next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.source import ContextFeatureSource, LogbookFeatureSource, MetadataFeatureSource


def window_index(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clip_id": [f"5401800{i}" for i in range(rows)],
            "tape_id": ["54018"] * rows,
            "species": ["SpermWhale"] * rows,
            "label": [1] * rows,
            "year": [1954] * rows,
            "native_sample_rate": [10000 + 100 * i for i in range(rows)],
            "duration_seconds": [1.5 + i for i in range(rows)],
            "bytes_on_disk": [30000 + i for i in range(rows)],
            "site": ["Bermuda", "Bermuda", "Cape Cod", "Cape Cod", "", "Bermuda"],
            "latitude": [32.3, 32.3, 41.7, 41.7, None, 32.3],
            "longitude": [-64.8, -64.8, -70.0, -70.0, None, -64.8],
            "collection_code": ["BA2A", "BA2A", "BE3C", "BE3C", "", "BA2A"],
            "cond_ship_noise": [True, False, False, True, False, False],
            "cond_reverberation": [False, True, False, False, True, False],
        }
    )


CONDITIONS = ["cond_ship_noise", "cond_reverberation"]


def test_the_context_control_sees_the_circumstances_of_a_recording():
    """Pinned because nine call type margins are distances from this.

    The collection code joined the list once it turned out to identify the species
    better than the audio did. Changing this list changes those nine numbers, which
    is the point of asserting it here.
    """
    source = ContextFeatureSource(window_index(), CONDITIONS)

    assert source.name == "context"
    assert source.feature_names() == [
        "site_code",
        "collection_code",
        "latitude",
        "longitude",
        "cond_ship_noise",
        "cond_reverberation",
    ]


def test_the_context_control_codes_names_and_fills_missing_coordinates():
    matrix = ContextFeatureSource(window_index(), CONDITIONS).matrix(np.arange(6)).to_numpy()

    assert matrix.shape == (6, 6)
    assert matrix[0, 0] == matrix[1, 0] == matrix[5, 0], "one site is one code"
    assert matrix[0, 0] != matrix[2, 0], "two sites are two codes"
    assert matrix[4, 2] == 0.0 and matrix[4, 3] == 0.0, "a missing coordinate reads as zero"
    assert matrix[0, 4] == 1.0 and matrix[1, 4] == 0.0


def test_the_metadata_control_sees_exactly_what_it_always_saw():
    """Pinned because every published species margin is a distance from this."""
    source = MetadataFeatureSource(window_index())

    assert source.name == "metadata"
    assert source.feature_names() == [
        "native_sample_rate",
        "year",
        "duration_seconds",
        "bytes_on_disk",
    ]
    assert source.matrix(np.arange(6)).to_numpy().shape == (6, 4)


def test_the_logbook_sees_the_paperwork_the_other_two_split_between_them():
    source = LogbookFeatureSource(window_index(), CONDITIONS)

    assert source.name == "logbook"
    assert source.feature_names() == [
        "site_code",
        "collection_code",
        "native_sample_rate",
        "year",
        "duration_seconds",
        "bytes_on_disk",
        "latitude",
        "longitude",
        "cond_ship_noise",
        "cond_reverberation",
    ]
    assert source.matrix(np.arange(6)).to_numpy().shape == (6, 10)


def test_the_logbook_codes_the_collection_as_an_identity():
    matrix = LogbookFeatureSource(window_index(), CONDITIONS).matrix(np.arange(6)).to_numpy()
    codes = matrix[:, 1]

    assert codes[0] == codes[1] == codes[5], "one collection is one code"
    assert codes[0] != codes[2], "two collections are two codes"
    assert len(set(codes)) == 3, "an absent code is its own category, not a missing value"


def test_the_logbook_strictly_contains_both_of_the_narrower_controls():
    """The point of it: one floor rather than two that each miss something."""
    logbook = set(LogbookFeatureSource(window_index(), CONDITIONS).feature_names())

    assert set(MetadataFeatureSource(window_index()).feature_names()) <= logbook
    assert set(ContextFeatureSource(window_index(), CONDITIONS).feature_names()) <= logbook


def test_a_control_refuses_an_index_that_cannot_support_it():
    thin = window_index().drop(columns=["collection_code"])
    with pytest.raises(ValueError, match="collection_code"):
        LogbookFeatureSource(thin, CONDITIONS)

    with pytest.raises(ValueError, match="site"):
        ContextFeatureSource(window_index().drop(columns=["site"]), CONDITIONS)
