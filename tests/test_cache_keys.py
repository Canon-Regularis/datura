"""What may and may not invalidate a feature cache.

Extraction over the real dataset takes minutes and the caches run to hundreds of
megabytes, so the key has to change for everything that alters the arrays and for
nothing else. Both halves of that are easy to get wrong in opposite directions, so
both are pinned here.
"""

from __future__ import annotations

from src.config import load_config
from tests.conftest import write_config


def _digests(tmp_path, name, **overrides):
    cfg = load_config(write_config(tmp_path / name, **overrides))
    return cfg.audio_digest, cfg.spectrogram_digest


def test_acquisition_details_do_not_invalidate_the_cache(tmp_path):
    """Pinning a checksum or switching mirror must not discard extracted features."""
    base = _digests(tmp_path, "base")
    moved = _digests(
        tmp_path,
        "moved",
        dataset={
            "archive_url": "https://mirror.invalid/somewhere-else.zip",
            "zip_name": "somewhere-else.zip",
            "archive_sha256": "a" * 64,
        },
    )
    assert base == moved


def test_species_selection_invalidates_the_cache(tmp_path):
    base = _digests(tmp_path, "base")
    fewer = _digests(tmp_path, "fewer", dataset={"species": ["HumpbackWhale", "SpermWhale"]})
    assert base != fewer


def test_audio_settings_invalidate_both_caches(tmp_path):
    base = _digests(tmp_path, "base")
    slower = _digests(
        tmp_path,
        "slower",
        audio={"target_sample_rate": 8000, "min_native_sample_rate": 8000},
        spectrogram={"fmax": 3900},
    )
    assert base[0] != slower[0]
    assert base[1] != slower[1]


def test_window_cap_invalidates_both_caches(tmp_path):
    base = _digests(tmp_path, "base")
    capped = _digests(tmp_path, "capped", audio={"max_windows_per_clip": 4})
    assert base[0] != capped[0]
    assert base[1] != capped[1]


def test_spectrogram_settings_leave_the_acoustic_cache_alone(tmp_path):
    """The mel grid changes the spectrograms. The descriptors are keyed separately
    so they survive a change to it."""
    base = _digests(tmp_path, "base")
    finer = _digests(tmp_path, "finer", spectrogram={"n_mels": 96})
    assert base[0] == finer[0]
    assert base[1] != finer[1]


def test_split_settings_do_not_touch_either_cache(tmp_path):
    """Folds are built after extraction, so reseeding them must not force a rebuild."""
    base = _digests(tmp_path, "base")
    reseeded = _digests(tmp_path, "reseeded", split={"n_folds": 4, "seed": 99})
    assert base == reseeded


def test_an_extractor_declares_which_settings_its_cache_depends_on(tmp_path):
    """The cache asks the extractor rather than testing its name.

    Choosing the digest by comparing against the literal "logmel" meant a second
    spectrogram derived representation would key off the audio settings alone, and
    changing the mel grid would hand it a stale array with no warning.
    """
    import numpy as np

    from src.features.base import FeatureExtractor
    from src.features.cache import cache_paths

    class Plain(FeatureExtractor):
        @property
        def name(self) -> str:
            return "plain"

        def output_shape(self, window_samples: int) -> tuple[int, ...]:
            return (4,)

        def transform(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

    class Melly(Plain):
        @property
        def name(self) -> str:
            return "melly"

        @property
        def cache_sections(self) -> tuple[str, ...]:
            return ("dataset", "audio", "spectrogram")

    base = load_config(write_config(tmp_path / "base"))
    finer = load_config(write_config(tmp_path / "finer", spectrogram={"n_mels": 96}))

    assert cache_paths(base, Plain())[0].name == cache_paths(finer, Plain())[0].name
    assert cache_paths(base, Melly())[0].name != cache_paths(finer, Melly())[0].name
