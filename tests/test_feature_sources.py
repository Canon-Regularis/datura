"""What each control is allowed to see, and in what order.

Every published margin is a distance from one of these. A column quietly added,
reordered or recoded moves numbers that are already written down, so the shape of
each control is pinned here rather than left to whoever edits it next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.controls import (
    LogbookFeatureSource,
    MetadataFeatureSource,
)


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


def test_the_logbook_gives_every_name_its_own_column():
    """Names are not numbers, and treating them as numbers cost a published claim.

    A site coded to its position in the alphabet is an ordinal, so the only question a
    split can ask is whether the code is above a threshold, and the answer for a name
    the model never saw is decided by where the alphabet put it. That is invisible
    under tape folds, where a held out tape almost always carries a name the training
    tapes carried too. Under place folds every held out name is new: five encodings
    carrying identical information scored between 0.7211 and 0.9993 on the same folds,
    and moving the absent name sentinel from one end of the axis to the other was worth
    0.278 on its own.
    """
    source = LogbookFeatureSource(window_index(), CONDITIONS)
    matrix = source.matrix(np.arange(6)).to_numpy()
    names = source.feature_names()

    assert matrix.shape[1] == len(names), "a column for every name and a name for every column"
    assert "site_Bermuda" in names and "site_Cape Cod" in names
    assert "collection_code_BA2A" in names

    bermuda = names.index("site_Bermuda")
    cape = names.index("site_Cape Cod")
    assert matrix[0, bermuda] == matrix[1, bermuda] == matrix[5, bermuda] == 1.0
    assert matrix[0, cape] == 0.0, "one name does not imply another"


def test_an_unrecorded_name_is_absent_rather_than_a_name_of_its_own():
    """Row four has no site and no collection code, and it must claim neither.

    Under place folds every held out name is unseen, so this is the state the model is
    in for the whole test set. Zero everywhere is the honest reading and it lets the
    tree fall back on what it does know.
    """
    source = LogbookFeatureSource(window_index(), CONDITIONS)
    matrix = source.matrix(np.arange(6)).to_numpy()
    names = source.feature_names()

    for prefix in ("site_", "collection_code_"):
        columns = [i for i, name in enumerate(names) if name.startswith(prefix)]
        assert matrix[4, columns].sum() == 0.0, f"row four claims a {prefix[:-1]} it has none of"
        assert matrix[0, columns].sum() == 1.0, "and a row that has one claims exactly one"
        assert not any(name == prefix.rstrip("_") for name in names), "absence gets no column"


def test_the_logbook_leaves_a_missing_number_missing():
    """This asserted that a missing coordinate reads as zero, and zero is a position.

    XGBoost takes NaN natively and learns a default direction per split, which is the
    honest reading of a value nobody wrote down. The fill was worth 0.010 under tape
    folds; what it was worth under place folds turned out to be a question about the
    name encoding above rather than about coordinates.
    """
    source = LogbookFeatureSource(window_index(), CONDITIONS)
    matrix = source.matrix(np.arange(6)).to_numpy()
    names = source.feature_names()

    latitude, longitude = names.index("latitude"), names.index("longitude")
    assert np.isnan(matrix[4, latitude]) and np.isnan(matrix[4, longitude])
    assert matrix[0, latitude] == pytest.approx(32.3), "a recorded one is untouched"


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
    """The numeric columns are pinned in order; the name columns are pinned by content.

    Their order follows the names present in the index, which differs between the
    species task and a call type subset, so pinning it would be pinning the fixture
    rather than the control.
    """
    source = LogbookFeatureSource(window_index(), CONDITIONS)
    names = source.feature_names()

    assert source.name == "logbook"
    assert [name for name in names if not name.startswith(("site_", "collection_code_"))] == [
        "native_sample_rate",
        "year",
        "duration_seconds",
        "bytes_on_disk",
        "latitude",
        "longitude",
        "cond_ship_noise",
        "cond_reverberation",
    ]
    assert set(names) >= {"site_Bermuda", "site_Cape Cod", "collection_code_BA2A"}
    assert source.matrix(np.arange(6)).to_numpy().shape == (6, len(names))


def test_the_logbook_reports_which_collection_rather_than_that_there_was_one():
    """A gain ranking can now name the code that carried a fit.

    It used to report a single ``collection_code`` column standing for all seven, so
    the importance tables could say the collection mattered and never which one.
    """
    source = LogbookFeatureSource(window_index(), CONDITIONS)
    matrix = source.matrix(np.arange(6)).to_numpy()
    names = source.feature_names()

    ba2a = names.index("collection_code_BA2A")
    be3c = names.index("collection_code_BE3C")
    assert list(matrix[:, ba2a]) == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert list(matrix[:, be3c]) == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_the_logbook_strictly_contains_the_metadata_control():
    """The point of it: one floor, and the narrower one says what paperwork adds."""
    logbook = set(LogbookFeatureSource(window_index(), CONDITIONS).feature_names())

    assert set(MetadataFeatureSource(window_index()).feature_names()) < logbook
    assert {"site_Bermuda", "collection_code_BA2A", "latitude", "longitude"} <= logbook


def test_a_control_refuses_an_index_that_cannot_support_it():
    thin = window_index().drop(columns=["collection_code"])
    with pytest.raises(ValueError, match="collection_code"):
        LogbookFeatureSource(thin, CONDITIONS)

    with pytest.raises(ValueError, match="site"):
        LogbookFeatureSource(window_index().drop(columns=["site"]), CONDITIONS)
