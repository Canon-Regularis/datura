"""The extractor interface.

Adding a new acoustic representation means adding a class here and registering it.
No trainer, cache or evaluation module needs to change, because they all work
against this interface rather than against a specific representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FeatureExtractor(ABC):
    """Turns one fixed-length window of audio into an array."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in cache filenames and reports."""

    @abstractmethod
    def output_shape(self, window_samples: int) -> tuple[int, ...]:
        """Shape produced for a single window, excluding the batch axis."""

    @abstractmethod
    def transform(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract from a single window."""

    def feature_names(self) -> list[str] | None:
        """Column names for flat representations, or None for image-shaped ones."""
        return None

    @property
    def storage_dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    def transform_batch(self, windows: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract from a ``(n_windows, window_samples)`` array."""
        if windows.ndim != 2:
            raise ValueError(f"expected a 2-D window array, got shape {windows.shape}")
        stacked = np.stack([self.transform(w, sample_rate) for w in windows])
        return stacked.astype(self.storage_dtype, copy=False)
