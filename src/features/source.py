"""Where a model's input matrix comes from.

The cross validation runner works against this interface, so cached spectrograms,
cached acoustic descriptors and every no-audio control flow through one code path
and produce directly comparable numbers.

Rows arrive as a lazy view rather than a materialised array; ``src.features.views``
owns that. The controls that read the window index rather than the audio live in
``src.features.controls``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.features.cache import FeatureStore
from src.features.views import RowView


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
    """Reads from a memory mapped cache produced by ``src.features.extract``."""

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


class DerivedSource(FeatureSource):
    """A view of another source over some of its windows, under different labels.

    The audio does not change between tasks, so neither should the cache. Asking
    whether a sperm whale clip contains a coda uses exactly the windows the species
    work already extracted; only the label and the selection differ.
    """

    def __init__(self, base: FeatureSource, index: pd.DataFrame, positions: np.ndarray, name: str):
        if len(index) != len(positions):
            raise ValueError(f"index has {len(index)} rows for {len(positions)} positions")
        self._base = base
        self._index = index.reset_index(drop=True)
        self._positions = np.asarray(positions, dtype=np.int64)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def matrix(self, rows: np.ndarray) -> RowView:
        return self._base.matrix(self._positions[np.asarray(rows, dtype=np.int64)])

    def feature_names(self) -> list[str] | None:
        return self._base.feature_names()
