"""The classifier interface.

Everything in ``src/train`` and ``src/evaluate`` accepts anything that satisfies
this, which is why the gradient boosted trees, the CNN and the metadata control
share one evaluation path. Adding a fourth model means adding a class here, not
editing the runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Batch:
    """Feature rows and their labels."""

    features: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"features ({len(self.features)}) and labels ({len(self.labels)}) differ in length"
            )

    def __len__(self) -> int:
        return len(self.labels)


def balanced_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Weights inversely proportional to class frequency.

    Clip counts run roughly one to three to six across the species set, and macro-F1
    is the reported metric, so the majority class must not dominate the loss.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    safe = np.where(counts > 0, counts, 1.0)
    weights = len(labels) / (n_classes * safe)
    return np.where(counts > 0, weights, 0.0)


class WindowClassifier(ABC):
    """Fits on windows and returns per-window class probabilities."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(self, train: Batch, validation: Batch, n_classes: int) -> None:
        """Fit on ``train``, using ``validation`` for early stopping only."""

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return a ``(n_windows, n_classes)`` array of probabilities."""

    @abstractmethod
    def save(self, path: Path) -> None: ...
