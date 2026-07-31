from __future__ import annotations

import numpy as np
import pytest

from src.audio.windows import pad_to_length, split_into_windows, window_starts


def test_short_signal_produces_one_window():
    assert window_starts(100, 200, 100) == [0]
    assert window_starts(200, 200, 100) == [0]


def test_regular_coverage():
    assert window_starts(500, 200, 100) == [0, 100, 200, 300]


def test_tail_is_covered_by_a_final_window():
    """A clip whose remainder is shorter than one hop still gets its end seen."""
    starts = window_starts(450, 200, 100)

    assert starts[-1] + 200 == 450
    assert starts == [0, 100, 200, 250]


def test_cap_thins_long_clips_evenly():
    uncapped = window_starts(10_000, 200, 100)
    capped = window_starts(10_000, 200, 100, max_windows=8)

    assert len(uncapped) > 8
    assert len(capped) == 8
    assert capped[0] == uncapped[0]
    assert capped[-1] == uncapped[-1]
    assert capped == sorted(capped)


def test_cap_is_inert_when_the_clip_is_short():
    assert window_starts(500, 200, 100, max_windows=16) == window_starts(500, 200, 100)


def test_padding_introduces_no_silence():
    signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    padded = pad_to_length(signal, 8, mode="reflect")

    assert len(padded) == 8
    assert np.count_nonzero(padded) == 8


def test_padding_falls_back_for_a_single_sample():
    padded = pad_to_length(np.array([0.5], dtype=np.float32), 4, mode="reflect")
    assert len(padded) == 4


def test_split_shapes():
    signal = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    windows = split_into_windows(signal, 200, 100)

    assert windows.shape == (len(window_starts(1000, 200, 100)), 200)
    assert windows.dtype == np.float32


def test_split_pads_a_clip_shorter_than_one_window():
    windows = split_into_windows(np.ones(50, dtype=np.float32), 200, 100)
    assert windows.shape == (1, 200)


@pytest.mark.parametrize("bad", [(0, 100), (200, 0)])
def test_rejects_non_positive_sizes(bad):
    with pytest.raises(ValueError):
        window_starts(1000, *bad)
