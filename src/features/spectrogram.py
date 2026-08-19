"""Log mel spectrograms, the learned representation input.

Each window is normalised on its own. That removes absolute recording level, which
varies with tape stock and gain staging rather than with the animal, so the CNN
cannot use loudness as a shortcut to the label.
"""

from __future__ import annotations

import librosa
import numpy as np

from src.config.sections import LOG_COMPRESSION, PCEN_COMPRESSION
from src.features.base import FeatureExtractor

_TOP_DB = 80.0

# See the note beside the same constant in acoustic.py.
_PCEN_SCALE = 2.0**31


class LogMelSpectrogram(FeatureExtractor):
    """Produces a ``(n_mels, n_frames)`` image per window."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        fmin: float,
        fmax: float,
        sample_rate: int,
        compression: str = LOG_COMPRESSION,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.compression = compression
        self._mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )

    @property
    def name(self) -> str:
        return "logmel"

    @property
    def cache_sections(self) -> tuple[str, ...]:
        return ("dataset", "audio", "spectrogram")

    @property
    def storage_dtype(self) -> np.dtype:
        # Values span roughly 80 dB after normalisation, far inside float16 range,
        # and halving the cache keeps the whole set memory mappable.
        return np.dtype(np.float16)

    def output_shape(self, window_samples: int) -> tuple[int, ...]:
        n_frames = 1 + window_samples // self.hop_length
        return (self.n_mels, n_frames)

    def mel_frequencies(self) -> np.ndarray:
        """Centre frequency of each mel band, used to label occlusion results."""
        return librosa.mel_frequencies(n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)

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
        if self.compression == PCEN_COMPRESSION:
            mel = librosa.pcen(
                (self._mel_basis @ magnitude) * _PCEN_SCALE,
                sr=self.sample_rate,
                hop_length=self.hop_length,
            )
        else:
            mel = librosa.power_to_db(self._mel_basis @ magnitude**2, ref=np.max, top_db=_TOP_DB)
        centred = mel - mel.mean()
        spread = centred.std()
        normalised = centred / spread if spread > 1e-6 else centred
        return normalised.astype(np.float32)
