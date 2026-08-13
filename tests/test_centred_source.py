"""Subtracting the recording, and leaving the animal alone.

The audio in this corpus identifies which of 56 tapes a clip came from at 0.816 macro-F1
where guessing reaches 0.018, and it recovers a tape's original sample rate after every
clip has been resampled to one rate. A per recording signature is present, and the place
held out result is what it costs.

``CentredSource`` removes the part of it that is a constant offset. What it must not do
is remove anything else, so the arithmetic is pinned here rather than left to the score
that motivated it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.source import CentredSource, FeatureSource
from src.features.views import RowView


class Fake(FeatureSource):
    """Rows in memory, so the transform is checked against arithmetic anybody can do."""

    def __init__(self, index: pd.DataFrame, block: np.ndarray):
        self._index = index.reset_index(drop=True)
        self._block = np.asarray(block, dtype=np.float32)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def matrix(self, rows: np.ndarray) -> RowView:
        return RowView.over(self._block[np.asarray(rows, dtype=np.int64)])

    def feature_names(self) -> list[str] | None:
        return ["a", "b"]


def two_tapes() -> Fake:
    """Two recordings whose offsets differ and whose shape around them does not.

    Tape one sits at (10, 20) and tape two at (100, 200). Within each, the windows are
    the same four corners. Centring should leave two identical sets of corners.
    """
    index = pd.DataFrame(
        {
            "clip_id": [f"1000{i}" for i in range(4)] + [f"2000{i}" for i in range(4)],
            "tape_id": ["10000"] * 4 + ["20000"] * 4,
            "label": [0] * 4 + [1] * 4,
        }
    )
    corners = np.array([[-1.0, -2.0], [1.0, 2.0], [-1.0, 2.0], [1.0, -2.0]])
    block = np.vstack([corners + np.array([10.0, 20.0]), corners + np.array([100.0, 200.0])])
    return Fake(index, block)


def test_each_recording_ends_up_centred_on_zero():
    """Which is the whole transform, and the reason a channel offset cannot survive it."""
    source = CentredSource(two_tapes(), name="centred")
    rows = np.arange(8)
    block = source.matrix(rows).take(rows)

    for start in (0, 4):
        np.testing.assert_allclose(block[start : start + 4].mean(axis=0), [0.0, 0.0], atol=1e-6)


def test_two_recordings_of_the_same_thing_become_the_same_rows():
    """The offset is all that separated them, so removing it has to collapse them.

    This is the mechanism the place experiment needs. Two tapes carrying identical
    within recording structure at different levels look like different data to a tree
    until the level is gone.
    """
    source = CentredSource(two_tapes(), name="centred")
    block = source.matrix(np.arange(8)).take(np.arange(8))

    np.testing.assert_allclose(block[:4], block[4:], atol=1e-6)


def test_the_spread_inside_a_recording_is_untouched():
    """Deliberate, and measured: dividing it out costs 0.111 under both fold rules.

    The per recording spread carries the animal, so only the mean goes. Compared within
    each recording rather than across both, because the spread across both is exactly
    the offset this removes.
    """
    source = CentredSource(two_tapes(), name="centred")
    before = two_tapes().matrix(np.arange(8)).take(np.arange(8))
    after = source.matrix(np.arange(8)).take(np.arange(8))

    for start in (0, 4):
        np.testing.assert_allclose(
            before[start : start + 4].std(axis=0), after[start : start + 4].std(axis=0), atol=1e-6
        )


def test_a_subset_of_rows_gets_its_own_recording_mean():
    """Rows arrive a fold at a time, so the mean has to follow the row rather than the call.

    Centring on whatever happens to be in the batch would make a model's input depend on
    which fold it landed in, and the training and test rows of one tape would be centred
    on different points.
    """
    source = CentredSource(two_tapes(), name="centred")
    everything = source.matrix(np.arange(8)).take(np.arange(8))

    for rows in (np.array([0, 5]), np.array([7]), np.array([3, 2, 6])):
        np.testing.assert_allclose(source.matrix(rows).take(np.arange(len(rows))), everything[rows])


def test_the_index_and_the_feature_names_pass_straight_through():
    """It is the same windows and the same descriptors, at a different offset."""
    base = two_tapes()
    source = CentredSource(base, name="centred")

    assert source.name == "centred"
    assert source.feature_names() == base.feature_names()
    pd.testing.assert_frame_equal(source.index, base.index)


def test_an_index_with_no_recording_column_is_refused():
    """Centring on the wrong thing is worse than not centring, so it is not guessed at."""
    thin = two_tapes()
    thin._index = thin._index.drop(columns=["tape_id"])  # noqa: SLF001 - no public way to thin it

    with pytest.raises(ValueError, match="tape_id"):
        CentredSource(thin, name="centred")


def test_the_registry_hands_it_out_under_a_name_of_its_own():
    """A derived representation is resolved like any other, and extracts nothing.

    It reads the parent's cache, so the extraction stage has nothing to do for it and
    ``kinds`` leaves it out. A name in ``kinds`` would make the pipeline try to build a
    cache that will never exist.
    """
    from src.features import registry

    assert registry.ACOUSTIC_CENTRED not in registry.kinds()
    assert registry.ACOUSTIC in registry.kinds()
