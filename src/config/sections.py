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
    corpus: str = ""
    """Which corpus this configuration reads, when it is not its own.

    A configuration names an experiment. The corpus is the data that experiment runs
    on, and two experiments can share one. ``context_10k`` differs from ``base_10k``
    only in what a fold boundary means, so it reads the same manifest, the same audit
    tables and the same feature caches rather than building duplicates of all three.

    Empty means the configuration owns its corpus, which is the usual case.
    """

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


LOG_COMPRESSION = "log"
PCEN_COMPRESSION = "pcen"
COMPRESSIONS = frozenset({LOG_COMPRESSION, PCEN_COMPRESSION})


@dataclass(frozen=True)
class SpectrogramConfig:
    n_fft: int
    hop_length: int
    n_mels: int
    fmin: float
    fmax: float
    # How mel energy is compressed. "log" is decibels against the window peak.
    # "pcen" divides by a smoothed estimate of itself instead, which suppresses
    # whatever is stationary in a recording and leaves what is transient.
    compression: str = LOG_COMPRESSION

    def __post_init__(self) -> None:
        if self.n_fft <= 0 or self.hop_length <= 0 or self.n_mels <= 0:
            raise ConfigError("spectrogram sizes must be positive")
        if self.hop_length > self.n_fft:
            raise ConfigError("spectrogram.hop_length must not exceed spectrogram.n_fft")
        if self.fmin >= self.fmax:
            raise ConfigError("spectrogram.fmin must be below spectrogram.fmax")
        if self.compression not in COMPRESSIONS:
            raise ConfigError(
                f"spectrogram.compression must be one of {sorted(COMPRESSIONS)}, "
                f"got {self.compression!r}"
            )

    @property
    def cache_identity(self) -> dict[str, Any]:
        """The settings that change the array, with the default compression left out.

        Every cache and every committed result in this project was built under log
        compression, and naming it here would move all three digests and orphan the
        features behind them. They would be rebuilt to exactly the bytes they already
        hold. So the default stays silent and only a departure from it is recorded,
        which is the same rule the dataset and encoder sections already follow.
        """
        identity: dict[str, Any] = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }
        if self.compression != LOG_COMPRESSION:
            identity["compression"] = self.compression
        return identity


@dataclass(frozen=True)
class EncoderConfig:
    """A pretrained audio encoder, used frozen to turn a window into a vector.

    Every other representation in this project is computed from the recording by code
    in this repository. This one arrives as weights, so the settings that decide what
    comes out of it are the architecture, the checkpoint behind it, and where the
    embedding is read from.

    ``sample_rate`` is what the encoder expects rather than what the corpus holds, and
    the two differ. The extractor resamples to meet it, which is the one place this
    project upsamples on purpose. See ``src.features.encoder`` for why that does not
    reintroduce the leak ``audio.min_native_sample_rate`` exists to prevent.
    """

    architecture: str = "wav2vec2_base"
    checkpoint: str = ""
    url: str = ""
    sha256: str = ""
    sample_rate: int = 16000
    layer: int = 1
    pooling: str = "mean"
    embedding_dim: int = 768
    batch_size: int = 32

    POOLINGS = ("mean", "max")

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.embedding_dim <= 0 or self.batch_size <= 0:
            raise ConfigError("encoder sample rate, embedding width and batch must be positive")
        if self.layer < 1:
            raise ConfigError("encoder.layer counts transformer layers from one")
        if self.pooling not in self.POOLINGS:
            raise ConfigError(f"encoder.pooling must be one of {list(self.POOLINGS)}")
        if self.sha256 and len(self.sha256) != 64:
            raise ConfigError("encoder.sha256 must be a 64 character hex digest")
        if self.checkpoint and not self.url:
            raise ConfigError("encoder.checkpoint names weights, so encoder.url must say where")

    @property
    def is_trained(self) -> bool:
        """Whether real weights back this, or it is a randomly initialised stand in.

        An empty checkpoint is how the synthetic corpus in the tests exercises the
        whole extraction path offline. Nothing under ``configs/`` may leave it empty.
        """
        return bool(self.checkpoint)

    @property
    def cache_identity(self) -> dict[str, Any]:
        """Only what changes the numbers that come out.

        The url and the digest describe where the weights were fetched from, in the
        same way the archive url describes acquisition rather than computation.
        Batching is excluded because a fixed length window makes it exact.
        """
        return {
            "architecture": self.architecture,
            "checkpoint": self.checkpoint,
            "sample_rate": self.sample_rate,
            "layer": self.layer,
            "pooling": self.pooling,
            "embedding_dim": self.embedding_dim,
        }


@dataclass(frozen=True)
class SplitConfig:
    n_folds: int
    seed: int
    tape_id_length: int
    group_column: str

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ConfigError("split.n_folds must be at least 2")
        if self.tape_id_length <= 0:
            raise ConfigError("split.tape_id_length must be positive")
        if not self.group_column:
            raise ConfigError("split.group_column must name a column in the manifest")


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
class PipelineConfig:
    """Which models a full pipeline run is allowed to train.

    ``None`` on either field means the default, which is what the two band variants
    want: every registered model, and call types inside the two species that carry
    enough of them. The wide species set declares less, because at eleven classes
    the networks cost hours of GPU to answer a question about the floor and its
    committed results are trees. A run that trained them anyway would rewrite a
    committed report with sections that were never meant to be in it.
    """

    models: tuple[str, ...] | None = None
    call_types: tuple[str, ...] | None = None

    def allows(self, model_name: str) -> bool:
        return self.models is None or model_name in self.models

    def call_type_species(self, default: tuple[str, ...]) -> tuple[str, ...]:
        return default if self.call_types is None else self.call_types


@dataclass(frozen=True)
class Config:
    name: str
    dataset: DatasetConfig
    audio: AudioConfig
    spectrogram: SpectrogramConfig
    split: SplitConfig
    paths: PathsConfig
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pipeline: PipelineConfig = field(compare=False, default_factory=PipelineConfig)
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
    def corpus(self) -> str:
        """The name the manifest, audits and feature caches are stored under.

        Distinct from ``name``, which is the experiment and decides where results go.
        """
        return self.dataset.corpus or self.name

    @property
    def audio_digest(self) -> str:
        return self.digest("dataset", "audio")

    @property
    def spectrogram_digest(self) -> str:
        return self.digest("dataset", "audio", "spectrogram")

    @property
    def encoder_digest(self) -> str:
        return self.digest("dataset", "audio", "encoder")
