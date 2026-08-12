"""Hand engineered acoustic descriptors, the interpretable baseline.

One short time Fourier transform is computed per window and every descriptor is
derived from it, which keeps extraction to a single transform instead of eight.

Recording metadata is deliberately absent. Native sample rate, year, file size and
clip duration all separate these species well and none of them describe the animal,
so they belong to the metadata control model where their contribution is measured
rather than hidden inside an "audio" result.
"""

from __future__ import annotations

import librosa
import numpy as np

from src.features.base import FeatureExtractor

_AGGREGATIONS = ("mean", "std", "p10", "p90")
_LOW_PERCENTILE = 10
_HIGH_PERCENTILE = 90
_CONTRAST_FMIN = 100.0


def contrast_band_count(sample_rate: int, fmin: float = _CONTRAST_FMIN) -> int:
    """Largest octave band count that fits below Nyquist.

    Spectral contrast doubles its band edges, so the usable count depends on the
    target rate. At 10 kHz five bands fit, at 5120 Hz only four.
    """
    nyquist = sample_rate / 2.0
    bands = 0
    while fmin * (2 ** (bands + 1)) < nyquist:
        bands += 1
    return max(bands, 1)


def _aggregate(series: np.ndarray) -> np.ndarray:
    """Collapse a ``(n_series, n_frames)`` block to four statistics per series.

    The two percentiles are computed separately on purpose. Asking numpy for both at
    once partitions the array a single time and is measurably faster, but on real
    MFCC blocks it picks a different representative among near equal values, and the
    result moves by one float32 step. That is enough to invalidate the cached
    features and every model fitted on them, which is a poor trade for two minutes
    off a step that runs once.
    """
    return np.concatenate(
        [
            series.mean(axis=1),
            series.std(axis=1),
            np.percentile(series, _LOW_PERCENTILE, axis=1),
            np.percentile(series, _HIGH_PERCENTILE, axis=1),
        ]
    )


def _aggregated_names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{i}_{stat}" for stat in _AGGREGATIONS for i in range(count)]


class AcousticFeatures(FeatureExtractor):
    """MFCCs, spectral shape descriptors, energy and rate measures."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        fmin: float,
        fmax: float,
        sample_rate: int,
        n_mfcc: int = 20,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_contrast_bands = contrast_band_count(sample_rate)
        self._mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        self._frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    @property
    def name(self) -> str:
        return "acoustic"

    @property
    def cache_sections(self) -> tuple[str, ...]:
        """The spectrogram belongs here, and its absence was a live bug.

        These descriptors are not spectrograms, so the section looks irrelevant, and
        it is not: the mel basis above is built from ``n_fft``, ``n_mels``, ``fmin``
        and ``fmax``, and every MFCC and contrast measure is computed through it.
        Moving ``n_mels`` from 64 to 96 changes 160 of the 214 values.

        What made it dangerous is that the count does not change. The output shape is
        the length of the feature names, which depends only on ``n_mfcc`` and the
        contrast band count, so a stale array loads at exactly the right shape and
        nothing raises. Editing a mel setting and rerunning the pipeline would have
        rebuilt the spectrograms and the embeddings, skipped these, and left the trees
        and both controls fitted on the old descriptors while the report claimed the
        new settings.
        """
        return ("dataset", "audio", "spectrogram")

    def output_shape(self, window_samples: int) -> tuple[int, ...]:
        return (len(self.feature_names()),)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        names += _aggregated_names("mfcc", self.n_mfcc)
        names += _aggregated_names("mfcc_delta", self.n_mfcc)
        names += _aggregated_names("contrast", self.n_contrast_bands + 1)
        for scalar in (
            "centroid",
            "bandwidth",
            "rolloff85",
            "flatness",
            "zero_crossing_rate",
            "rms",
            "spectral_entropy",
        ):
            names += [f"{scalar}_{stat}" for stat in _AGGREGATIONS]
        names += ["dominant_frequency", "dominant_frequency_ratio"]
        return names

    def transform(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"extractor was built for {self.sample_rate} Hz but received {sample_rate} Hz"
            )

        magnitude = np.abs(
            librosa.stft(
                np.asarray(window, dtype=np.float32),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
        )
        power = magnitude**2
        mel_power = self._mel_basis @ power
        mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=80.0)

        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc, width=min(9, _odd_below(mfcc.shape[1])))

        contrast = librosa.feature.spectral_contrast(
            S=magnitude,
            sr=sample_rate,
            fmin=_CONTRAST_FMIN,
            n_bands=self.n_contrast_bands,
        )

        centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)
        # Bandwidth is measured about the centroid, and librosa recomputes it unless
        # it is handed one. Passing it saves a second pass over the spectrogram.
        bandwidth = librosa.feature.spectral_bandwidth(
            S=magnitude, sr=sample_rate, centroid=centroid
        )
        rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate, roll_percent=0.85)
        flatness = librosa.feature.spectral_flatness(S=magnitude)
        zero_crossing = librosa.feature.zero_crossing_rate(
            window, frame_length=self.n_fft, hop_length=self.hop_length
        )
        rms = librosa.feature.rms(S=magnitude, frame_length=self.n_fft, hop_length=self.hop_length)
        entropy = _spectral_entropy(power)

        mean_spectrum = power.mean(axis=1)
        peak = int(np.argmax(mean_spectrum))
        total = float(mean_spectrum.sum())
        dominant = np.array(
            [
                self._frequencies[peak],
                float(mean_spectrum[peak]) / total if total > 0 else 0.0,
            ],
            dtype=np.float64,
        )

        blocks = [
            _aggregate(mfcc),
            _aggregate(mfcc_delta),
            _aggregate(contrast),
            _aggregate(centroid),
            _aggregate(bandwidth),
            _aggregate(rolloff),
            _aggregate(flatness),
            _aggregate(zero_crossing),
            _aggregate(rms),
            _aggregate(entropy),
            dominant,
        ]
        vector = np.concatenate(blocks).astype(np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _odd_below(n_frames: int) -> int:
    """Largest odd width a delta filter can use for this many frames."""
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    return max(width, 3)


def _spectral_entropy(power: np.ndarray) -> np.ndarray:
    """Per frame Shannon entropy of the normalised power spectrum, in [0, 1].

    Separates tonal calls, whose energy sits in few bins, from broadband clicks,
    whose energy spreads across the spectrum.
    """
    total = power.sum(axis=0, keepdims=True)
    distribution = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(distribution > 0, np.log(distribution), 0.0)
    entropy = -(distribution * logs).sum(axis=0) / np.log(power.shape[0])
    return entropy[np.newaxis, :]
