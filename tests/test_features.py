from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import registry
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


def test_a_cache_refuses_to_be_written_with_clips_missing(config, monkeypatch, caplog):
    """A short cache is not visibly wrong later, and it breaks a shared fold set.

    The probe and the networks are comparable because their two caches cover the same
    clips. If one extractor silently dropped a file the other kept, two models would
    be scored on different corpora under the same fold numbers.
    """
    from src.features.extract import ExtractionIncomplete, extract

    manifest = pd.DataFrame(
        {
            "clip_id": ["a", "b"],
            "tape_id": ["t0000", "t0000"],
            "species": ["HumpbackWhale", "HumpbackWhale"],
            "label": [0, 0],
            "year": [1970, 1970],
            "native_sample_rate": [10000, 10000],
            "duration_seconds": [2.0, 2.0],
            "bytes_on_disk": [1000, 1000],
            "relative_path": ["missing_one.wav", "missing_two.wav"],
        }
    )
    extractor = registry.build_extractor(registry.ACOUSTIC, config)

    with pytest.raises(ExtractionIncomplete, match="2 of 2"):
        extract(config, extractor, manifest)

    record = config.paths.metadata / f"extract_failures_{config.name}_acoustic.csv"
    assert record.exists(), "the clips that failed have to be named somewhere"
    assert set(pd.read_csv(record)["clip_id"]) == {"a", "b"}


def test_a_cache_can_be_built_without_the_missing_clips_on_request(config):
    """The escape hatch, so a corrupt file does not stop the world without a decision."""
    from src.features.extract import extract

    manifest = pd.DataFrame(
        {
            "clip_id": ["a"],
            "tape_id": ["t0000"],
            "species": ["HumpbackWhale"],
            "label": [0],
            "year": [1970],
            "native_sample_rate": [10000],
            "duration_seconds": [2.0],
            "bytes_on_disk": [1000],
            "relative_path": ["missing.wav"],
        }
    )
    extractor = registry.build_extractor(registry.ACOUSTIC, config)
    store = extract(config, extractor, manifest, allow_failures=True)
    assert len(store.index) == 0
