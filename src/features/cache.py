"""On disk cache for extracted features.

Cache filenames carry a hash of every configuration section that affects the
arrays, so changing the target sample rate or the mel settings produces a new key
and a stale cache can never be silently reused against a new configuration.

Windows are written out as they are produced rather than gathered in a list, which
keeps peak memory flat regardless of how much audio the species set covers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.features.base import FeatureExtractor


@dataclass(frozen=True)
class FeatureStore:
    """A feature array and the window index describing its rows."""

    features: np.ndarray
    index: pd.DataFrame

    def __post_init__(self) -> None:
        if len(self.features) != len(self.index):
            raise ValueError(
                f"feature rows ({len(self.features)}) and index rows ({len(self.index)}) differ"
            )


def cache_paths(cfg: Config, extractor: FeatureExtractor) -> tuple[Path, Path]:
    """Array path and index path for this configuration and extractor.

    The digest comes from whichever config sections the extractor says it depends
    on, so this module never needs to know one representation from another. Testing
    the name here instead would mean a new spectrogram derived representation
    silently reused a cache built under different mel settings.
    """
    stem = f"{cfg.name}_{extractor.name}_{cfg.digest(*extractor.cache_sections)}"
    return (
        cfg.paths.processed / f"{stem}.npy",
        cfg.paths.processed / f"{stem}_index.parquet",
    )


def exists(cfg: Config, extractor: FeatureExtractor) -> bool:
    array_path, index_path = cache_paths(cfg, extractor)
    return array_path.exists() and index_path.exists()


class FeatureWriter:
    """Streams feature blocks to disk, then seals them into a .npy file."""

    def __init__(
        self, cfg: Config, extractor: FeatureExtractor, shape: tuple[int, ...], dtype: np.dtype
    ):
        self.array_path, self.index_path = cache_paths(cfg, extractor)
        self.array_path.parent.mkdir(parents=True, exist_ok=True)
        self._scratch = self.array_path.with_suffix(".partial")
        self._handle = self._scratch.open("wb")
        self._shape = tuple(shape)
        self._dtype = np.dtype(dtype)
        self._rows = 0

    def append(self, block: np.ndarray) -> int:
        """Write a ``(n_windows, *shape)`` block and return how many rows were added."""
        if block.shape[1:] != self._shape:
            raise ValueError(f"expected rows shaped {self._shape}, got {block.shape[1:]}")
        self._handle.write(np.ascontiguousarray(block, dtype=self._dtype).tobytes())
        self._rows += len(block)
        return len(block)

    def close(self, index: pd.DataFrame) -> FeatureStore:
        self._handle.close()
        if len(index) != self._rows:
            raise ValueError(f"wrote {self._rows} rows but the index describes {len(index)}")

        flat = np.memmap(self._scratch, dtype=self._dtype, mode="r")
        array = flat.reshape((self._rows, *self._shape))
        np.save(self.array_path, array)
        del flat, array
        self._scratch.unlink()

        index.to_parquet(self.index_path, index=False)
        return load(self.array_path, self.index_path)


def load(array_path: Path, index_path: Path) -> FeatureStore:
    return FeatureStore(
        features=np.load(array_path, mmap_mode="r"),
        index=pd.read_parquet(index_path),
    )


def load_cached(cfg: Config, extractor: FeatureExtractor) -> FeatureStore:
    array_path, index_path = cache_paths(cfg, extractor)
    if not (array_path.exists() and index_path.exists()):
        raise FileNotFoundError(
            f"no cached {extractor.name} features for config {cfg.name}; "
            f"run python -m src.features.extract --config {cfg.source.name}"
        )
    return load(array_path, index_path)
