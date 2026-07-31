from __future__ import annotations

import numpy as np
import pytest

from src.features.acoustic import AcousticFeatures, contrast_band_count
from src.features.spectrogram import LogMelSpectrogram

RATE = 10_000
WINDOW = 20_000


def tone(frequency: float, samples: int = WINDOW, rate: int = RATE) -> np.ndarray:
    time = np.arange(samples, dtype=np.float32) / rate
    return np.sin(2 * np.pi * frequency * time).astype(np.float32)


def clicks(interval: float = 0.4, samples: int = WINDOW, rate: int = RATE) -> np.ndarray:
    signal = np.zeros(samples, dtype=np.float32)
    signal[:: int(interval * rate)] = 1.0
    return signal


@pytest.fixture
def acoustic() -> AcousticFeatures:
    return AcousticFeatures(
        n_fft=512, hop_length=64, n_mels=64, fmin=50, fmax=4900, sample_rate=RATE
    )


@pytest.fixture
def logmel() -> LogMelSpectrogram:
    return LogMelSpectrogram(
        n_fft=512, hop_length=64, n_mels=64, fmin=50, fmax=4900, sample_rate=RATE
    )


def test_contrast_bands_fit_below_nyquist():
    assert contrast_band_count(10_000) == 5
    assert contrast_band_count(5_120) == 4


def test_vector_length_matches_declared_names(acoustic):
    vector = acoustic.transform(tone(800), RATE)

    assert vector.shape == acoustic.output_shape(WINDOW)
    assert len(vector) == len(acoustic.feature_names())
    assert len(set(acoustic.feature_names())) == len(acoustic.feature_names())


def test_features_are_finite_on_silence_and_noise(acoustic):
    for signal in (
        np.zeros(WINDOW, dtype=np.float32),
        np.random.default_rng(0).standard_normal(WINDOW).astype(np.float32),
        clicks(),
    ):
        vector = acoustic.transform(signal, RATE)
        assert np.isfinite(vector).all()


def test_extraction_is_deterministic(acoustic):
    signal = tone(1200)
    assert np.array_equal(acoustic.transform(signal, RATE), acoustic.transform(signal, RATE))


def test_dominant_frequency_finds_the_tone(acoustic):
    names = acoustic.feature_names()
    position = names.index("dominant_frequency")
    vector = acoustic.transform(tone(1500), RATE)

    assert abs(vector[position] - 1500) < 60


def test_entropy_separates_tonal_from_broadband_energy(acoustic):
    """The descriptor that should distinguish a whistle from a click train."""
    names = acoustic.feature_names()
    position = names.index("spectral_entropy_mean")

    tonal = acoustic.transform(tone(1000), RATE)[position]
    broadband = acoustic.transform(
        np.random.default_rng(0).standard_normal(WINDOW).astype(np.float32), RATE
    )[position]

    assert broadband > 2 * tonal


def test_rejects_a_mismatched_sample_rate(acoustic):
    with pytest.raises(ValueError, match="built for"):
        acoustic.transform(tone(500), 44_100)


def test_spectrogram_shape_matches_its_declaration(logmel):
    image = logmel.transform(tone(600), RATE)

    assert image.shape == logmel.output_shape(WINDOW)
    assert image.shape[0] == 64


def test_spectrogram_is_level_invariant(logmel):
    quiet = logmel.transform(tone(600) * 0.01, RATE)
    loud = logmel.transform(tone(600) * 1.0, RATE)

    assert np.allclose(quiet, loud, atol=1e-3)


def test_spectrogram_batch_uses_the_storage_dtype(logmel):
    windows = np.stack([tone(400), tone(1600)])
    block = logmel.transform_batch(windows, RATE)

    assert block.shape == (2, *logmel.output_shape(WINDOW))
    assert block.dtype == np.float16


def test_mel_frequencies_stay_inside_the_band(logmel):
    frequencies = logmel.mel_frequencies()

    assert len(frequencies) == 64
    assert frequencies[0] >= 50
    assert frequencies[-1] <= 4900
