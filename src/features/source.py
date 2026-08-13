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


class CentredSource(FeatureSource):
    """Another source with each recording's mean feature vector subtracted.

    A stationary recording channel adds roughly the same offset to every window of a
    tape, so the tape mean is largely equipment and the variation around it is largely
    animal. Subtracting the mean is cepstral mean normalisation, which speaker
    recognition has used for the same reason for decades.

    It is worth 0.077 macro-F1 under tape folds and 0.180 under place folds, and the
    second being more than double the first is the point: what it removes mattered most
    when travelling between recording locations. Also dividing by the per recording
    spread costs 0.111 under both fold rules, so that spread carries the animal and is
    left alone.

    Grouping is always the recording, never the fold's grouping column. Centring by
    place under place folds would pool statistics across every tape at a location, which
    is a different transform and a much leakier one.

    Nothing is written to the cache. The means are one vector per recording and the
    subtraction happens on read, so this costs a pass over the parent at construction
    and a copy of whichever rows a caller asks for.
    """

    GROUP = "tape_id"

    def __init__(self, base: FeatureSource, name: str):
        if self.GROUP not in base.index.columns:
            raise ValueError(f"window index carries no {self.GROUP} to centre within")
        self._base = base
        self._name = name
        self._means: dict[object, np.ndarray] = {}
        self._of_row = base.index[self.GROUP].to_numpy()
        for group, positions in base.index.groupby(self.GROUP).indices.items():
            rows = np.asarray(positions, dtype=np.int64)
            block = np.asarray(base.matrix(rows).take(np.arange(len(rows))), dtype=np.float32)
            self._means[group] = block.mean(axis=0)

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> pd.DataFrame:
        return self._base.index

    def matrix(self, rows: np.ndarray) -> RowView:
        wanted = np.asarray(rows, dtype=np.int64)
        block = np.asarray(self._base.matrix(wanted).take(np.arange(len(wanted))), dtype=np.float32)
        offsets = np.stack([self._means[group] for group in self._of_row[wanted]])
        return RowView.over(block - offsets)

    def feature_names(self) -> list[str] | None:
        return self._base.feature_names()
