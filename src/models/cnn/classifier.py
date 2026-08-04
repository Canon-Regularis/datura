"""Fitting the network, and predicting with it.

The training loop, the learning rate schedule and the batching sit here. The
architecture, the augmentation and the device flags are imported; this module is
about the procedure rather than the parts.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.features.views import RowView
from src.models.base import Batch, FoldContext, WindowClassifier, balanced_class_weights
from src.models.cnn.augment import SpectrogramAugment
from src.models.cnn.network import MelResNet
from src.models.cnn.runtime import configure_backend, resolve_device


def _batches(
    view: RowView, labels: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    order = rng.permutation(len(view)) if shuffle else np.arange(len(view))
    for start in range(0, len(order), batch_size):
        positions = order[start : start + batch_size]
        yield view.take(positions), labels[positions]


class SpectrogramCNN(WindowClassifier):
    def __init__(
        self,
        model_settings: dict[str, Any],
        train_settings: dict[str, Any],
        augment_settings: dict[str, Any],
    ):
        self._model_settings = dict(model_settings)
        self._train_settings = dict(train_settings)
        self._augment = SpectrogramAugment(augment_settings)
        self.device = resolve_device(str(train_settings.get("device", "auto")))
        self.module: MelResNet | None = None
        self.history: list[dict[str, float]] = []

    @property
    def name(self) -> str:
        return "cnn"

    def _build(self, n_classes: int) -> MelResNet:
        return MelResNet(n_classes=n_classes, **self._model_settings).to(self.device)

    def fit(self, train: Batch, validation: Batch, n_classes: int) -> None:
        seed = int(self._train_settings.get("seed", 0))
        configure_backend(self.device, bool(self._train_settings.get("deterministic", False)))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        self.module = self._build(n_classes)
        weights = balanced_class_weights(train.labels, n_classes)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

        epochs = int(self._train_settings.get("epochs", 40))
        batch_size = int(self._train_settings.get("batch_size", 32))
        warmup = int(self._train_settings.get("warmup_epochs", 3))
        patience = int(self._train_settings.get("early_stopping_patience", 8))
        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=float(self._train_settings.get("lr", 3e-3)),
            weight_decay=float(self._train_settings.get("weight_decay", 0.01)),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: _schedule(epoch, warmup, epochs)
        )
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler(self.device.type, enabled=use_amp)

        best_score = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0

        for epoch in range(epochs):
            self.module.train()
            total_loss = 0.0
            seen = 0
            for features, labels in _batches(train.features, train.labels, batch_size, True, rng):
                inputs = self._to_tensor(features)
                inputs = self._augment(inputs, generator)
                targets = torch.as_tensor(labels, dtype=torch.long, device=self.device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type, enabled=use_amp):
                    loss = criterion(self.module(inputs), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.detach().item() * len(labels)
                seen += len(labels)
            scheduler.step()

            probabilities = self.predict_proba(validation.features)
            score = f1_score(
                validation.labels, probabilities.argmax(axis=1), average="macro", zero_division=0
            )
            self.history.append(
                {"epoch": epoch, "train_loss": total_loss / max(seen, 1), "val_macro_f1": score}
            )

            if score > best_score:
                best_score, stale = score, 0
                best_state = copy.deepcopy(self.module.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(1) if tensor.ndim == 3 else tensor

    @torch.no_grad()
    def predict_proba(self, features: RowView) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("fit must be called before predict_proba")
        self.module.eval()
        batch_size = int(self._train_settings.get("batch_size", 32)) * 2
        outputs = []
        for start in range(0, len(features), batch_size):
            positions = np.arange(start, min(start + batch_size, len(features)))
            logits = self.module(self._to_tensor(features.take(positions)))
            outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    def artifacts(self, context: FoldContext) -> dict[str, pd.DataFrame]:
        """The learning curve, and the weights that produced it.

        Saving here keeps the checkpoint beside the fold it belongs to, so the
        explainability tools load exactly the model that was scored.
        """
        self.save(context.checkpoint)
        return {"history": pd.DataFrame(self.history)}

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


def _schedule(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base rate."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
