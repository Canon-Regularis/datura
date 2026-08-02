"""Posing a call type question inside one species.

The risks here are quiet ones. A task can look fine on clip counts while resting on
two recordings; a relabelled view can drift out of step with the rows it points at.
Both would produce a number that reads as a result and is not one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.notes import CALL_PREFIX
from src.features.source import ContextFeatureSource, DerivedSource, FeatureSource, RowView
from src.train.calltypes import (
    ABSENT,
    PRESENT,
    CallTypeError,
    Task,
    _window_index,
    viable_tasks,
)


class FakeSource(FeatureSource):
    """A source backed by an array whose value encodes its own row number."""

    def __init__(self, index: pd.DataFrame):
        self._index = index.reset_index(drop=True)
        self._matrix = np.arange(len(self._index), dtype=np.float32).reshape(-1, 1)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def matrix(self, rows: np.ndarray) -> RowView:
        return RowView(self._matrix, rows)

    def feature_names(self) -> list[str]:
        return ["row"]


def build_labels(tapes_per_type: dict[str, int], clips_per_tape: int = 8) -> pd.DataFrame:
    """One clip row per cut, with a call type flag column per type."""
    rows = []
    for type_index, (call_type, tapes) in enumerate(tapes_per_type.items()):
        for tape in range(tapes):
            tape_id = f"{type_index}{tape:04d}"
            for cut in range(clips_per_tape):
                row = {
                    "clip_id": f"{tape_id}{cut:03d}",
                    "tape_id": tape_id,
                    "species": "SpermWhale",
                    "site": f"site {type_index}",
                    "latitude": 10.0 + type_index,
                    "longitude": -20.0 - type_index,
                    "cond_ship_noise": cut % 2 == 0,
                }
                row.update({f"{CALL_PREFIX}{t}": t == call_type for t in tapes_per_type})
                rows.append(row)
    return pd.DataFrame(rows)


def test_a_call_type_on_too_few_tapes_is_refused():
    """Clip count is not sample size. Humpback song spans 95 clips and 2 tapes."""
    labels = build_labels({"click": 12, "song": 2}, clips_per_tape=40)

    tasks = viable_tasks(None, "SpermWhale", labels)

    assert [task.call_type for task in tasks] == ["click"]
    assert tasks[0].tapes == 12


def test_no_viable_call_type_is_an_error_rather_than_an_empty_run():
    labels = build_labels({"song": 2}, clips_per_tape=40)
    with pytest.raises(CallTypeError, match="reaches"):
        viable_tasks(None, "HumpbackWhale", labels)


def test_tasks_are_ordered_by_independent_recordings():
    labels = build_labels({"click": 14, "coda": 22, "whistle": 11})
    tasks = viable_tasks(None, "SpermWhale", labels)
    assert [task.call_type for task in tasks] == ["coda", "click", "whistle"]


def test_relabelling_marks_the_right_windows_present():
    labels = build_labels({"coda": 12, "click": 12})
    windows = labels.loc[labels.index.repeat(3)].reset_index(drop=True)
    source = FakeSource(windows[["clip_id", "tape_id", "species"]].assign(label=0))

    subset, positions = _window_index(source, labels, Task("SpermWhale", "coda", 96, 12))

    assert len(subset) == len(windows)
    assert set(subset["species"]) == {PRESENT, ABSENT}

    coda_clips = set(labels[labels[f"{CALL_PREFIX}coda"]]["clip_id"])
    marked = set(subset[subset["label"] == 1]["clip_id"])
    assert marked == coda_clips
    assert len(positions) == len(subset)


def test_a_derived_view_reads_the_rows_it_points_at():
    """The value in each fake row is its own base position, so drift is visible."""
    index = pd.DataFrame(
        {
            "clip_id": [f"c{i}" for i in range(10)],
            "tape_id": ["t0"] * 10,
            "species": ["X"] * 10,
            "label": [0] * 10,
        }
    )
    base = FakeSource(index)
    positions = np.array([3, 5, 7])
    derived = DerivedSource(base, index.iloc[positions], positions, name="derived")

    taken = derived.matrix(np.array([0, 1, 2])).to_numpy().ravel()
    np.testing.assert_array_equal(taken, positions)

    single = derived.matrix(np.array([2])).to_numpy().ravel()
    np.testing.assert_array_equal(single, [7])


def test_a_derived_view_refuses_an_index_that_does_not_match_its_rows():
    index = pd.DataFrame(
        {"clip_id": ["a", "b"], "tape_id": ["t", "t"], "species": ["X", "X"], "label": [0, 1]}
    )
    with pytest.raises(ValueError, match="positions"):
        DerivedSource(FakeSource(index), index, np.array([0]), name="bad")


def test_the_context_control_sees_place_and_conditions_but_no_audio():
    labels = build_labels({"coda": 12, "click": 12})
    control = ContextFeatureSource(labels, ["cond_ship_noise"])

    assert control.feature_names() == ["site_code", "latitude", "longitude", "cond_ship_noise"]

    matrix = control.matrix(np.arange(len(labels))).to_numpy()
    assert matrix.shape == (len(labels), 4)
    # Two sites in this fixture, so the code takes exactly two values.
    assert len(np.unique(matrix[:, 0])) == 2


def test_the_context_control_refuses_an_index_without_a_site():
    frame = pd.DataFrame({"latitude": [1.0], "longitude": [2.0]})
    with pytest.raises(ValueError, match="context columns"):
        ContextFeatureSource(frame, [])
