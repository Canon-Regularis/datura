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
import pandas as pd

from src.features.views import RowView


@dataclass(frozen=True)
class FoldContext:
    """What a model may need beyond its training data, once per fold.

    Passed to ``artifacts`` so each model can report on itself without the runner
    knowing what kind of model it is holding.
    """

    fold_index: int
    feature_names: list[str] | None
    checkpoint: Path


@dataclass(frozen=True)
class Batch:
    """Feature rows and their labels.

    A ``RowView`` rather than an array, because that is what every caller has always
    passed. Declaring an array here was a lie the implementations then worked around
    individually, and one of them worked around it with ``isinstance``, which is how a
    lookalike object reached numpy as a nought dimensional object array.
    """

    features: RowView
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
    """Fits on windows and returns per window class probabilities."""

    @abstractmethod
    def fit(self, train: Batch, validation: Batch, n_classes: int) -> None:
        """Fit on ``train``, using ``validation`` for early stopping only."""

    @abstractmethod
    def predict_proba(self, features: RowView) -> np.ndarray:
        """Return a ``(n_windows, n_classes)`` array of probabilities.

        A view names its rows and materialises them in blocks, so a model that only
        needs one batch at a time never holds a whole fold in memory. Ask it for
        ``to_numpy`` or ``take`` rather than testing what it is.
        """

    @abstractmethod
    def save(self, path: Path) -> None: ...

    def artifacts(self, context: FoldContext) -> dict[str, pd.DataFrame]:
        """Tables this model can report about one fold of itself.

        Trees return which features carried the fit; the network returns its
        learning curve and writes its weights. A model with nothing to add returns
        nothing, which is why this is not abstract.
        """
        return {}
