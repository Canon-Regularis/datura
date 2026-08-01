"""What a configuration is, and what makes one invalid.

Each section validates itself the moment it is constructed, so an impossible
combination fails at load time rather than halfway through a training run. Nothing
here reads a file: these are the shapes, and ``loading`` fills them in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.errors import DaturaError

PAD_MODES = {"reflect", "wrap", "edge", "constant"}


class ConfigError(DaturaError, ValueError):
    """Raised when a configuration file is missing keys or internally inconsistent."""


@dataclass(frozen=True)
class DatasetConfig:
    archive_url: str
    zip_name: str
    archive_sha256: str
    archive_root: str
    species: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.archive_sha256) != 64 or not all(
            c in "0123456789abcdef" for c in self.archive_sha256
        ):
            raise ConfigError("dataset.archive_sha256 must be a 64 character hex digest")
        if not self.species:
            raise ConfigError("dataset.species must list at least one species")
        if len(set(self.species)) != len(self.species):
            raise ConfigError("dataset.species contains duplicates")

    @property
    def label_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.species)}

    @property
    def cache_identity(self) -> dict[str, Any]:
        """Only the fields that change what gets extracted.

        Where the audio came from, its filename and its checksum describe the
        acquisition rather than the computation. Folding them into the cache key
        would mean pinning a checksum or switching mirror throws away hundreds of
        megabytes of perfectly valid features.
        """
        return {"species": list(self.species)}


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int
    min_native_sample_rate: int
    window_seconds: float
    hop_seconds: float
    min_clip_seconds: float
    pad_mode: str
    max_windows_per_clip: int

    def __post_init__(self) -> None:
        if self.target_sample_rate <= 0:
            raise ConfigError("audio.target_sample_rate must be positive")
        if self.min_native_sample_rate < self.target_sample_rate:
            raise ConfigError(
                "audio.min_native_sample_rate must be at least audio.target_sample_rate, "
                "otherwise files get upsampled and gain an empty high band that leaks the label"
            )
        if self.window_seconds <= 0 or self.hop_seconds <= 0:
            raise ConfigError("audio.window_seconds and audio.hop_seconds must be positive")
        if self.hop_seconds > self.window_seconds:
            raise ConfigError("audio.hop_seconds must not exceed audio.window_seconds")
        if self.min_clip_seconds <= 0:
            raise ConfigError("audio.min_clip_seconds must be positive")
        if self.pad_mode not in PAD_MODES:
            raise ConfigError(f"audio.pad_mode must be one of {sorted(PAD_MODES)}")
        if self.max_windows_per_clip < 0:
            raise ConfigError("audio.max_windows_per_clip must be zero or positive")

    @property
    def window_samples(self) -> int:
        return round(self.window_seconds * self.target_sample_rate)

    @property
    def hop_samples(self) -> int:
        return round(self.hop_seconds * self.target_sample_rate)

    @property
    def nyquist(self) -> float:
        return self.target_sample_rate / 2.0


@dataclass(frozen=True)
class SpectrogramConfig:
    n_fft: int
    hop_length: int
    n_mels: int
    fmin: float
    fmax: float

    def __post_init__(self) -> None:
        if self.n_fft <= 0 or self.hop_length <= 0 or self.n_mels <= 0:
            raise ConfigError("spectrogram sizes must be positive")
        if self.hop_length > self.n_fft:
            raise ConfigError("spectrogram.hop_length must not exceed spectrogram.n_fft")
        if self.fmin >= self.fmax:
            raise ConfigError("spectrogram.fmin must be below spectrogram.fmax")


@dataclass(frozen=True)
class SplitConfig:
    n_folds: int
    seed: int
    tape_id_length: int

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ConfigError("split.n_folds must be at least 2")
        if self.tape_id_length <= 0:
            raise ConfigError("split.tape_id_length must be positive")


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    metadata: Path
    processed: Path
    reports: Path

    def ensure(self) -> None:
        for path in (self.raw, self.metadata, self.processed, self.reports):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    name: str
    dataset: DatasetConfig
    audio: AudioConfig
    spectrogram: SpectrogramConfig
    split: SplitConfig
    paths: PathsConfig
    source: Path = field(compare=False, default=Path())

    def __post_init__(self) -> None:
        if self.spectrogram.fmax > self.audio.nyquist:
            raise ConfigError(
                f"spectrogram.fmax ({self.spectrogram.fmax}) exceeds the Nyquist frequency "
                f"of audio.target_sample_rate ({self.audio.nyquist})"
            )
        if self.spectrogram.n_fft > self.audio.window_samples:
            raise ConfigError(
                "spectrogram.n_fft is longer than one analysis window; "
                "shorten n_fft or lengthen audio.window_seconds"
            )

    def digest(self, *sections: str) -> str:
        """Stable short hash of the named sections, used to key cached artifacts.

        Changing any setting that affects a derived array changes its cache key, so a
        stale cache can never be silently reused against a new configuration.
        """
        payload: dict[str, Any] = {}
        for section in sections:
            value = getattr(self, section)
            identity = getattr(value, "cache_identity", None)
            if identity is not None:
                payload[section] = identity
            else:
                payload[section] = (
                    asdict(value) if hasattr(value, "__dataclass_fields__") else value
                )
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def audio_digest(self) -> str:
        return self.digest("dataset", "audio")

    @property
    def spectrogram_digest(self) -> str:
        return self.digest("dataset", "audio", "spectrogram")
