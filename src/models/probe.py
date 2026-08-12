"""A small head over frozen encoder embeddings.

The encoder is not fitted here, or anywhere. It runs once, its output is cached like
any other representation, and this trains the only part that moves. That is what
makes a fifty split run affordable, and it is also what makes the comparison clean:
the probe and the networks are scored on identical folds of identical recordings.

The head is deliberately small. A linear map over a pretrained representation is the
standard way to ask what that representation already separates, and adding capacity
here would start answering a different question, about what can be learned from 134
recordings rather than about what the encoder brought with it. ``hidden: 0`` gives
the bare linear probe; anything larger inserts one hidden layer.

Standardisation is fitted on the training rows of each fold and carried in the
checkpoint. Fitting it on everything would leak the test fold's scale into training,
which is a small leak and still a leak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.base import Batch
from src.models.torch_base import TorchWindowClassifier


def _head(width: int, hidden: int, dropout: float, n_classes: int) -> nn.Module:
    """Linear, or one hidden layer when asked for."""
    if hidden <= 0:
        return nn.Linear(width, n_classes)
    return nn.Sequential(
        nn.Linear(width, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


class EmbeddingProbe(TorchWindowClassifier):
    """One head, fitted per fold, over vectors the encoder already produced."""

    def __init__(self, model_settings: dict[str, Any], train_settings: dict[str, Any]):
        super().__init__(train_settings)
        self._model_settings = dict(model_settings)
        self.centre: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "probe"

    def _prepare(self, train: Batch) -> None:
        """Centre and scale from the training rows of this fold alone.

        Before the head is built, because its width is the width of what it is about
        to see. Fitting this on everything would leak the test fold's scale into
        training, which is a small leak and still a leak.
        """
        rows = np.asarray(train.features.take(np.arange(len(train.features))), dtype=np.float32)
        centre = rows.mean(axis=0)
        spread = rows.std(axis=0)
        # A constant column carries nothing and must not become an infinity.
        spread[spread < 1e-6] = 1.0
        self.centre = torch.as_tensor(centre, device=self.device)
        self.scale = torch.as_tensor(spread, device=self.device)

    def _build(self, n_classes: int) -> nn.Module:
        return _head(
            int(self.centre.shape[0]),
            int(self._model_settings["hidden"]),
            float(self._model_settings["dropout"]),
            n_classes,
        ).to(self.device)

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=self.device)
        if tensor.ndim != 2:
            raise ValueError(f"the probe expects one vector per window, got {tuple(tensor.shape)}")
        if self.centre is None or self.scale is None:
            raise RuntimeError("fit must be called before the probe standardises anything")
        return (tensor - self.centre) / self.scale

    def save(self, path: Path) -> None:
        if self.module is None:
            raise RuntimeError("fit must be called before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.module.state_dict(),
                "settings": self._model_settings,
                "centre": self.centre.cpu(),
                "scale": self.scale.cpu(),
            },
            path.with_suffix(".pt"),
        )

    @classmethod
    def load(cls, path: Path, train_settings: dict[str, Any], n_classes: int) -> EmbeddingProbe:
        """Rebuild a fitted probe, standardisation and all."""
        checkpoint = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        model = cls(checkpoint["settings"], train_settings)
        model.centre = checkpoint["centre"].to(model.device)
        model.scale = checkpoint["scale"].to(model.device)
        model.module = _head(
            int(model.centre.shape[0]),
            int(checkpoint["settings"]["hidden"]),
            float(checkpoint["settings"]["dropout"]),
            n_classes,
        ).to(model.device)
        model.module.load_state_dict(checkpoint["state_dict"])
        model.module.eval()
        return model
