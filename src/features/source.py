"""Where a model's input matrix comes from.

The cross validation runner works against this interface, so cached spectrograms,
cached acoustic descriptors and the metadata control all flow through one code path
and produce directly comparable numbers.

Rows are handed out as a lazy view rather than a materialised array. A fold of
log mel windows runs to hundreds of megabytes and the cache is memory mapped, so
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


class TabularFeatureSource(FeatureSource):
    """A control built from columns of the window index, with no audio.

    Every control in this project is the same shape: some names coded as identities
    for a tree to split on, some numbers taken as they are, and nothing that came
    out of the recording itself. Written once so the three of them cannot drift
    apart in how they treat a missing value or a category.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        *,
        name: str,
        categorical: list[str],
        numeric: list[str],
    ):
        missing = {*categorical, *numeric} - set(index.columns)
        if missing:
            raise ValueError(f"window index is missing {name} columns: {sorted(missing)}")

        self._index = index.reset_index(drop=True)
        self._name = name
        self._categorical = list(categorical)
        self._numeric = list(numeric)

        # A name is coded rather than measured. Trees split on the code as an
        # identity, which is all a control needs it to be. Nothing is filled in
        # first: an absent name codes to -1, which is a category of its own and is
        # what the context control has always done.
        codes = [
            self._index[column].astype("category").cat.codes.to_numpy()
            for column in self._categorical
        ]
        numbers = self._index.loc[:, self._numeric].astype(float).fillna(0.0).to_numpy()
        self._matrix = np.column_stack([*codes, numbers]).astype(np.float32)

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def matrix(self, rows: np.ndarray) -> RowView:
        return RowView(self._matrix, rows)

    def feature_names(self) -> list[str]:
        # A coded column says so in its name, and one that already says so is left
        # alone. The context control has reported "site_code" since it was written.
        coded = [
            column if column.endswith("_code") else f"{column}_code" for column in self._categorical
        ]
        return coded + self._numeric


class ContextFeatureSource(TabularFeatureSource):
    """Where a recording was made and what else was audible, with no audio.

    The control for a call type task. Site, coordinates, the collection a cut came
    from and the noise conditions describe the circumstances of a recording rather
    than the animal, so whatever this reaches is the floor an audio model has to
    clear.

    The collection code was added after it turned out to identify the species better
    than the audio did. A call type task is posed inside one species so the code
    cannot give the species away, but several collections sit inside each species and
    a recordist who taped one behaviour on one trip would leave the same shape of
    trace. Measuring it is cheaper than assuming it is absent.
    """

    BASE_COLUMNS = ("latitude", "longitude")

    def __init__(self, index: pd.DataFrame, condition_columns: list[str]):
        super().__init__(
            index,
            name="context",
            categorical=["site", "collection_code"],
            numeric=[*self.BASE_COLUMNS, *condition_columns],
        )


class LogbookFeatureSource(TabularFeatureSource):
    """Everything written down about a recording, and none of the recording.

    The other two controls each miss something. The species control never sees the
    site or the collection a cut came from; the call type control never sees the
    sample rate or the year. This one sees the lot, so it is the floor rather than
    a floor, and the gap between it and the narrower controls says how much each
    piece of paperwork was worth.

    The collection code earns its place here on measurement. Three codes carry 98%
    of the clips under study, each sits in exactly one species, and they span 61, 51
    and 12 tapes, so a fold boundary drawn between tapes does nothing to hide them.
    """

    CATEGORICAL = ("site", "collection_code")
    NUMERIC = (
        "native_sample_rate",
        "year",
        "duration_seconds",
        "bytes_on_disk",
        "latitude",
        "longitude",
    )

    def __init__(self, index: pd.DataFrame, condition_columns: list[str]):
        super().__init__(
            index,
            name="logbook",
            categorical=list(self.CATEGORICAL),
            numeric=[*self.NUMERIC, *condition_columns],
        )


class MetadataFeatureSource(FeatureSource):
    """Recording metadata only, with no audio content whatsoever.

    This is the control. Native sample rate, recording year, clip duration and file
    size describe the tape and the equipment. None of them describe the animal.
    Whatever accuracy this reaches is the floor an audio model has to clear before its
    score can be read as evidence about whale vocalisation.
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
