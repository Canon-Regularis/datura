"""Handing a model its rows without copying them.

A fold of log mel windows runs to hundreds of megabytes and the cache is memory
mapped, so materialising a whole fold to pass it to a model would be the largest
allocation in the project for no benefit. A view names the rows instead, and copies
only the block a model asks for.

Nothing here knows what a feature means. That is what lets the occlusion test hide a
frequency band by wrapping a view, without touching the model or the cache.
"""

from __future__ import annotations

import numpy as np


class RowView:
    """A selection of rows from a backing array, materialised only in blocks."""

    def __init__(self, array: np.ndarray, rows: np.ndarray):
        self._array = array
        self._rows = np.asarray(rows, dtype=np.int64)

    @classmethod
    def over(cls, array: np.ndarray) -> RowView:
        """Every row of an array already in memory.

        One file has no cache behind it, so the prediction command has an array and
        needs a view. It was building ``RowView(matrix, np.arange(len(matrix)))`` by
        hand, which is the kind of line that gets copied wrong once and then reaches
        numpy as something that is not a view at all.
        """
        return cls(array, np.arange(len(array)))

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
