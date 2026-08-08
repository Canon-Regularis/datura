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

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.features.views import RowView
from src.models.base import Batch, FoldContext, WindowClassifier, balanced_class_weights
from src.models.runtime import batches, breathe, configure_backend, resolve_device, schedule


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


class EmbeddingProbe(WindowClassifier):
    """One head, fitted per fold, over vectors the encoder already produced."""

    def __init__(self, model_settings: dict[str, Any], train_settings: dict[str, Any]):
        self._model_settings = dict(model_settings)
        self._train_settings = dict(train_settings)
        self.device = resolve_device(str(train_settings.get("device", "auto")))
        self.module: nn.Module | None = None
        self.centre: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.history: list[dict[str, float]] = []

    @property
    def name(self) -> str:
        return "probe"

    def _standardise_on(self, view: RowView) -> None:
        """Centre and scale from the training rows of this fold alone."""
        rows = np.asarray(view.take(np.arange(len(view))), dtype=np.float32)
        centre = rows.mean(axis=0)
        spread = rows.std(axis=0)
        # A constant column carries nothing and must not become an infinity.
        spread[spread < 1e-6] = 1.0
        self.centre = torch.as_tensor(centre, device=self.device)
        self.scale = torch.as_tensor(spread, device=self.device)

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=self.device)
        if tensor.ndim != 2:
            raise ValueError(f"the probe expects one vector per window, got {tuple(tensor.shape)}")
        if self.centre is None or self.scale is None:
            raise RuntimeError("fit must be called before the probe standardises anything")
        return (tensor - self.centre) / self.scale

    def fit(self, train: Batch, validation: Batch, n_classes: int) -> None:
        seed = int(self._train_settings.get("seed", 0))
        configure_backend(self.device, bool(self._train_settings.get("deterministic", False)))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        self._standardise_on(train.features)
        width = int(self.centre.shape[0])
        self.module = _head(
            width,
            int(self._model_settings.get("hidden", 0)),
            float(self._model_settings.get("dropout", 0.0)),
            n_classes,
        ).to(self.device)

        weights = balanced_class_weights(train.labels, n_classes)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

        epochs = int(self._train_settings.get("epochs", 60))
        batch_size = int(self._train_settings.get("batch_size", 256))
        warmup = int(self._train_settings.get("warmup_epochs", 3))
        patience = int(self._train_settings.get("early_stopping_patience", 10))
        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=float(self._train_settings.get("lr", 1e-3)),
            weight_decay=float(self._train_settings.get("weight_decay", 0.01)),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: schedule(epoch, warmup, epochs)
        )

        best_score = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0

        for epoch in range(epochs):
            self.module.train()
            total_loss = 0.0
            seen = 0
            for features, labels in batches(train.features, train.labels, batch_size, True, rng):
                inputs = self._to_tensor(features)
                targets = torch.as_tensor(labels, dtype=torch.long, device=self.device)

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.module(inputs), targets)
                loss.backward()
                optimizer.step()

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

            breathe()

            if score > best_score:
                best_score, stale = score, 0
                best_state = copy.deepcopy(self.module.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)

    @torch.no_grad()
    def predict_proba(self, features: RowView) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("fit must be called before predict_proba")
        self.module.eval()
        batch_size = int(self._train_settings.get("batch_size", 256)) * 2
        outputs = []
        for start in range(0, len(features), batch_size):
            positions = np.arange(start, min(start + batch_size, len(features)))
            logits = self.module(self._to_tensor(features.take(positions)))
            outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    def artifacts(self, context: FoldContext) -> dict[str, pd.DataFrame]:
        self.save(context.checkpoint)
        return {"history": pd.DataFrame(self.history)}

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
            int(checkpoint["settings"].get("hidden", 0)),
            float(checkpoint["settings"].get("dropout", 0.0)),
            n_classes,
        ).to(model.device)
        model.module.load_state_dict(checkpoint["state_dict"])
        model.module.eval()
        return model
