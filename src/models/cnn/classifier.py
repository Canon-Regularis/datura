"""Fitting the network, and predicting with it.

What is specific to this model and nothing else: the architecture to build, how a
batch of log mel windows becomes a tensor, and the augmentation applied on the way in.
The epoch loop, the schedule, the early stopping and the learning curve are the same
procedure the probe uses and live in ``src.models.torch_base``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.cnn.augment import SpectrogramAugment
from src.models.cnn.network import MelResNet
from src.models.torch_base import TorchWindowClassifier


class SpectrogramCNN(TorchWindowClassifier):
    """A residual network over log mel windows."""

    # Mixed precision on a GPU, which is most of why a fold takes seventeen minutes
    # rather than half an hour. Disabled on CPU by the base, so a CI run is unaffected.
    USES_AMP = True

    def __init__(
        self,
        model_settings: dict[str, Any],
        train_settings: dict[str, Any],
        augment_settings: dict[str, Any],
    ):
        super().__init__(train_settings)
        self._model_settings = dict(model_settings)
        self._augmentation = SpectrogramAugment(augment_settings)

    @property
    def name(self) -> str:
        return "cnn"

    def _build(self, n_classes: int) -> MelResNet:
        return MelResNet(n_classes=n_classes, **self._model_settings).to(self.device)

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(1) if tensor.ndim == 3 else tensor

    def _augment(self, inputs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        return self._augmentation(inputs, generator)

    def save(self, path: Path) -> None:
        if self.module is None:
            raise RuntimeError("fit must be called before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.module.state_dict(), "settings": self._model_settings},
            path.with_suffix(".pt"),
        )

    @classmethod
    def load(cls, path: Path, train_settings: dict[str, Any], n_classes: int) -> SpectrogramCNN:
        """Rebuild a fitted model from a checkpoint, for explainability runs."""
        checkpoint = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        model = cls(checkpoint["settings"], train_settings, {"enabled": False})
        model.module = model._build(n_classes)
        model.module.load_state_dict(checkpoint["state_dict"])
        model.module.eval()
        return model
