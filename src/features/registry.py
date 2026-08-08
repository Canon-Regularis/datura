"""The catalogue of acoustic representations.

Adding a representation means writing the extractor and adding one line here.
Nothing else in the project names a representation directly, so the trainers, the
cache and the explainability tools all pick it up without being edited.
"""

from __future__ import annotations

from collections.abc import Callable

from src.config import Config
from src.errors import DaturaError
from src.features import cache
from src.features.acoustic import AcousticFeatures
from src.features.base import FeatureExtractor
from src.features.encoder import EncoderEmbedding
from src.features.source import CachedFeatureSource
from src.features.spectrogram import LogMelSpectrogram

ACOUSTIC = "acoustic"
LOGMEL = "logmel"
ENCODER = "encoder"


def _spectral(extractor: type[FeatureExtractor]) -> Callable[[Config], FeatureExtractor]:
    """The wiring the two hand written representations share."""

    def build(cfg: Config) -> FeatureExtractor:
        return extractor(
            n_fft=cfg.spectrogram.n_fft,
            hop_length=cfg.spectrogram.hop_length,
            n_mels=cfg.spectrogram.n_mels,
            fmin=cfg.spectrogram.fmin,
            fmax=cfg.spectrogram.fmax,
            sample_rate=cfg.audio.target_sample_rate,
        )

    return build


# A factory per representation rather than a class per representation. The two
# spectral extractors take the same six settings and the encoder takes none of them,
# so a single constructor call could not describe all three.
_EXTRACTORS: dict[str, Callable[[Config], FeatureExtractor]] = {
    ACOUSTIC: _spectral(AcousticFeatures),
    LOGMEL: _spectral(LogMelSpectrogram),
    ENCODER: EncoderEmbedding,
}


class UnknownRepresentation(DaturaError):
    """Raised when a representation name has no extractor behind it."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"unknown representation {kind!r}; expected one of {kinds()}")


def kinds() -> tuple[str, ...]:
    """Every representation the project can extract."""
    return tuple(_EXTRACTORS)


def build_extractor(kind: str, cfg: Config) -> FeatureExtractor:
    """Configure the extractor for one representation.

    Building one is cheap and side effect free for every representation, including
    the encoder, which defers its weights until the first window arrives. Callers
    that only want a cache path or a feature name can rely on that.
    """
    if kind not in _EXTRACTORS:
        raise UnknownRepresentation(kind)
    return _EXTRACTORS[kind](cfg)


def cache_exists(kind: str, cfg: Config) -> bool:
    """Whether one representation has already been extracted under this config.

    Callers name a representation; only the extractor knows which config sections
    its cache key is built from. Asking here keeps ``cache`` from having to import
    this module, which would be a cycle.
    """
    return cache.exists(cfg, build_extractor(kind, cfg))


def load_source(kind: str, cfg: Config) -> CachedFeatureSource:
    """Open the cached features for one representation, ready to train on.

    Feature names come from the extractor rather than being repeated here, so a
    report of which descriptors mattered can never drift out of step with the
    columns the model was actually given.
    """
    extractor = build_extractor(kind, cfg)
    return CachedFeatureSource(
        cache.load_cached(cfg, extractor),
        name=kind,
        feature_names=extractor.feature_names(),
    )
