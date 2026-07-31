"""Where a model's input matrix comes from.

The cross-validation runner works against this interface, so cached spectrograms,
cached acoustic descriptors and the metadata control all flow through one code path
and produce directly comparable numbers.

Rows are handed out as a lazy view rather than a materialised array. A fold of
log-mel windows runs to hundreds of megabytes and the cache is memory-mapped, so
copying a whole fold into RAM to hand it to a model would be the largest allocation
in the project for no benefit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.features.cache import FeatureStore


class RowView:
    """A selection of rows from a backing array, materialised only in blocks."""

    def __init__(self, array: np.ndarray, rows: np.ndarray):
        self._array = array
        self._rows = np.asarray(rows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._rows), *self._array.shape[1:])

    @property
    def row_shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape[1:])

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def rows(self) -> np.ndarray:
        return self._rows

    def take(self, positions: np.ndarray) -> np.ndarray:
        """Materialise the given positions within this view as float32."""
        return np.ascontiguousarray(self._array[self._rows[positions]], dtype=np.float32)

    def to_numpy(self) -> np.ndarray:
        """Materialise every row. Only safe for the flat representations."""
        return np.ascontiguousarray(self._array[self._rows], dtype=np.float32)


class MaskedRowView(RowView):
    """A row view with part of every row blanked out.

    Used by the occlusion test to hide a frequency band from a trained model
    without touching the model or the cache.
    """

    def __init__(self, base: RowView, mask: np.ndarray):
        super().__init__(base.array, base.rows)
        if mask.shape != base.row_shape:
            raise ValueError(f"mask shape {mask.shape} does not match rows {base.row_shape}")
        self._mask = mask.astype(np.float32)

    def take(self, positions: np.ndarray) -> np.ndarray:
        return super().take(positions) * self._mask

    def to_numpy(self) -> np.ndarray:
        return super().to_numpy() * self._mask


class FeatureSource(ABC):
    """A window index plus the matrix rows that go with it."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def index(self) -> pd.DataFrame:
        """One row per window, carrying clip_id, tape_id, species and label."""

    @abstractmethod
    def matrix(self, rows: np.ndarray) -> RowView:
        """A lazy view of the feature rows at the given positions."""

    def feature_names(self) -> list[str] | None:
        return None


class CachedFeatureSource(FeatureSource):
    """Reads from a memory-mapped cache produced by ``src.features.extract``."""

    def __init__(self, store: FeatureStore, name: str, feature_names: list[str] | None = None):
        self._store = store
        self._name = name
        self._feature_names = feature_names

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> pd.DataFrame:
        return self._store.index

    def matrix(self, rows: np.ndarray) -> RowView:
        return RowView(self._store.features, rows)

    def feature_names(self) -> list[str] | None:
        return self._feature_names


class MetadataFeatureSource(FeatureSource):
    """Recording metadata only, with no audio content whatsoever.

    This is the control. Native sample rate, recording year, clip duration and file
    size describe the tape and the equipment, not the animal. Whatever accuracy this
    reaches is the floor an audio model has to clear before its score can be read as
    evidence about whale vocalisation.
    """

    COLUMNS = ("native_sample_rate", "year", "duration_seconds", "bytes_on_disk")

    def __init__(self, index: pd.DataFrame):
        missing = set(self.COLUMNS) - set(index.columns)
        if missing:
            raise ValueError(f"window index is missing metadata columns: {sorted(missing)}")
        self._index = index.reset_index(drop=True)
        self._matrix = self._index.loc[:, list(self.COLUMNS)].to_numpy(dtype=np.float32)

    @property
    def name(self) -> str:
        return "metadata"

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def matrix(self, rows: np.ndarray) -> RowView:
        return RowView(self._matrix, rows)

    def feature_names(self) -> list[str]:
        return list(self.COLUMNS)
