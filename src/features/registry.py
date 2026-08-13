"""The catalogue of acoustic representations.

Adding a representation means writing the extractor and adding one line here.
Nothing else in the project names a representation directly, so the trainers, the
cache and the explainability tools all pick it up without being edited.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.config import Config
from src.data.annotations import condition_columns
from src.errors import DaturaError
from src.features import cache
from src.features.acoustic import AcousticFeatures
from src.features.base import FeatureExtractor
from src.features.controls import LogbookFeatureSource, MetadataFeatureSource
from src.features.encoder import EncoderEmbedding
from src.features.source import CachedFeatureSource, CentredSource, FeatureSource
from src.features.spectrogram import LogMelSpectrogram

ACOUSTIC = "acoustic"
LOGMEL = "logmel"
ENCODER = "encoder"

# Representations built by transforming another one rather than by reading audio. They
# read the parent's cache and write none of their own, so adding one costs a name here
# and nothing on disk. ``load_source`` resolves them; ``kinds`` does not list them,
# because there is nothing for the extraction stage to do.
_DERIVED: dict[str, str] = {}


def _derived(name: str, parent: str) -> str:
    _DERIVED[name] = parent
    return name


# The acoustic descriptors with each recording's mean subtracted. What it is worth, and
# why the per recording spread is deliberately left alone, is in ``CentredSource``.
ACOUSTIC_CENTRED = _derived("acoustic_centred", ACOUSTIC)

# The two that hear nothing. They are feature sources like any other and were keyed
# from a different module, with an if chain in a third resolving which was which, so
# ``ModelSpec.source`` was a union of two namespaces and adding a control meant editing
# three files. They are named here, beside the representations, because that is what
# makes the field one namespace.
METADATA = "metadata"
LOGBOOK = "logbook"


def _spectral(extractor: Callable[..., FeatureExtractor]) -> Callable[[Config], FeatureExtractor]:
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


# A control is derived from a window index rather than read from a cache, so it takes
# the frame instead of a config. Same registry, different shape of factory, which is
# why the two tables are separate rather than one with a discriminator.
_CONTROLS: dict[str, Callable[[pd.DataFrame], FeatureSource]] = {
    METADATA: MetadataFeatureSource,
    LOGBOOK: lambda index: LogbookFeatureSource(index, condition_columns(index)),
}


def controls() -> tuple[str, ...]:
    """Every source that describes a recording without hearing it."""
    return tuple(_CONTROLS)


def is_control(name: str) -> bool:
    return name in _CONTROLS


def build_control(name: str, index: pd.DataFrame) -> FeatureSource:
    """One control over the clips a window index names.

    Derived from the audio source's index so a control sees the same clips in the same
    folds and only the description differs.
    """
    if name not in _CONTROLS:
        raise UnknownRepresentation(name)
    return _CONTROLS[name](index)


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

    A derived representation has the cache of whatever it is derived from, so asking
    about it asks about the parent rather than reporting a cache that will never exist.
    """
    return cache.exists(cfg, build_extractor(_DERIVED.get(kind, kind), cfg))


def load_source(kind: str, cfg: Config) -> FeatureSource:
    """Open the cached features for one representation, ready to train on.

    Feature names come from the extractor rather than being repeated here, so a
    report of which descriptors mattered can never drift out of step with the
    columns the model was actually given.
    """
    if kind in _DERIVED:
        return CentredSource(load_source(_DERIVED[kind], cfg), name=kind)

    extractor = build_extractor(kind, cfg)
    return CachedFeatureSource(
        cache.load_cached(cfg, extractor),
        name=kind,
        feature_names=extractor.feature_names(),
    )
