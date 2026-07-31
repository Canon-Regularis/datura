from __future__ import annotations

import numpy as np
import pytest

from src.audio.resample import UpsamplingRejected, to_target_rate


def test_downsampling_halves_the_length():
    signal = np.sin(np.linspace(0, 40 * np.pi, 20_000)).astype(np.float32)
    converted = to_target_rate(signal, 20_000, 10_000)

    assert abs(len(converted) - 10_000) <= 1
    assert converted.dtype == np.float32


def test_matching_rate_is_a_pass_through():
    signal = np.linspace(-1, 1, 512, dtype=np.float32)
    assert np.array_equal(to_target_rate(signal, 10_000, 10_000), signal)


def test_upsampling_is_refused_rather_than_performed():
    """A 5120 Hz file stretched to 10 kHz has an empty band above 2560 Hz, and a
    classifier will read that emptiness as a species label."""
    signal = np.zeros(5120, dtype=np.float32)

    with pytest.raises(UpsamplingRejected) as raised:
        to_target_rate(signal, 5120, 10_000)

    assert raised.value.native_rate == 5120
    assert raised.value.target_rate == 10_000


def test_band_limiting_removes_content_above_the_new_nyquist():
    rate = 40_000
    time = np.arange(rate, dtype=np.float32) / rate
    signal = np.sin(2 * np.pi * 15_000 * time).astype(np.float32)

    converted = to_target_rate(signal, rate, 10_000)
    spectrum = np.abs(np.fft.rfft(converted))

    assert spectrum.max() < 0.05 * np.abs(np.fft.rfft(signal)).max()
