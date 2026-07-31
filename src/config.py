"""Single parse and validation point for YAML configuration.

Every other module receives an already-validated ``Config`` object. Nothing else
in the codebase reads a raw dict or reaches into the YAML file, so a typo or an
impossible combination of settings fails here rather than halfway through a
training run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PAD_MODES = {"reflect", "wrap", "edge", "constant"}


class ConfigError(ValueError):
    """Raised when a configuration file is missing keys or internally inconsistent."""


@dataclass(frozen=True)
class DatasetConfig:
    archive_url: str
    zip_name: str
    archive_root: str
    species: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.species:
            raise ConfigError("dataset.species must list at least one species")
        if len(set(self.species)) != len(self.species):
            raise ConfigError("dataset.species contains duplicates")

    @property
    def label_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.species)}


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
        if self.pad_mode not in _PAD_MODES:
            raise ConfigError(f"audio.pad_mode must be one of {sorted(_PAD_MODES)}")
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
            payload[section] = asdict(value) if hasattr(value, "__dataclass_fields__") else value
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def audio_digest(self) -> str:
        return self.digest("dataset", "audio")

    @property
    def spectrogram_digest(self) -> str:
        return self.digest("dataset", "audio", "spectrogram")


def _require(mapping: dict[str, Any], section: str, allowed: set[str]) -> dict[str, Any]:
    if section not in mapping:
        raise ConfigError(f"missing required config section: {section}")
    block = mapping[section]
    if not isinstance(block, dict):
        raise ConfigError(f"config section {section} must be a mapping")
    unknown = set(block) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in {section}: {sorted(unknown)}")
    missing = allowed - set(block)
    if missing:
        raise ConfigError(f"missing keys in {section}: {sorted(missing)}")
    return block


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> Config:
    """Read a YAML config, validate it, and return the typed object."""
    source = Path(path)
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()
    if not source.exists():
        raise ConfigError(f"config file not found: {source}")

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {source} must contain a top-level mapping")
    if "name" not in raw:
        raise ConfigError("config must define a top-level 'name'")

    dataset_block = _require(
        raw, "dataset", {"archive_url", "zip_name", "archive_root", "species"}
    )
    audio_block = _require(
        raw,
        "audio",
        {
            "target_sample_rate",
            "min_native_sample_rate",
            "window_seconds",
            "hop_seconds",
            "min_clip_seconds",
            "pad_mode",
            "max_windows_per_clip",
        },
    )
    spec_block = _require(raw, "spectrogram", {"n_fft", "hop_length", "n_mels", "fmin", "fmax"})
    split_block = _require(raw, "split", {"n_folds", "seed", "tape_id_length"})
    paths_block = _require(raw, "paths", {"raw", "metadata", "processed", "reports"})

    return Config(
        name=str(raw["name"]),
        dataset=DatasetConfig(
            archive_url=str(dataset_block["archive_url"]),
            zip_name=str(dataset_block["zip_name"]),
            archive_root=str(dataset_block["archive_root"]),
            species=tuple(str(s) for s in dataset_block["species"]),
        ),
        audio=AudioConfig(
            target_sample_rate=int(audio_block["target_sample_rate"]),
            min_native_sample_rate=int(audio_block["min_native_sample_rate"]),
            window_seconds=float(audio_block["window_seconds"]),
            hop_seconds=float(audio_block["hop_seconds"]),
            min_clip_seconds=float(audio_block["min_clip_seconds"]),
            pad_mode=str(audio_block["pad_mode"]),
            max_windows_per_clip=int(audio_block["max_windows_per_clip"]),
        ),
        spectrogram=SpectrogramConfig(
            n_fft=int(spec_block["n_fft"]),
            hop_length=int(spec_block["hop_length"]),
            n_mels=int(spec_block["n_mels"]),
            fmin=float(spec_block["fmin"]),
            fmax=float(spec_block["fmax"]),
        ),
        split=SplitConfig(
            n_folds=int(split_block["n_folds"]),
            seed=int(split_block["seed"]),
            tape_id_length=int(split_block["tape_id_length"]),
        ),
        paths=PathsConfig(
            raw=_resolve(paths_block["raw"]),
            metadata=_resolve(paths_block["metadata"]),
            processed=_resolve(paths_block["processed"]),
            reports=_resolve(paths_block["reports"]),
        ),
        source=source,
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a model hyperparameter file. Model configs stay plain dicts because
    their keys are passed straight through to the estimator they configure."""
    source = Path(path)
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()
    if not source.exists():
        raise ConfigError(f"config file not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config file {source} must contain a top-level mapping")
    return data
