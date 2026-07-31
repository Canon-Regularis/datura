"""Residual CNN over log-mel windows.

Small on purpose. The species set survives on a few dozen independent tapes, so
capacity is not the limiting factor and a larger network would only memorise tapes
faster. Four stages at base width 32 comes to roughly two million parameters.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.features.source import RowView
from src.models.base import Batch, WindowClassifier, balanced_class_weights


def resolve_device(requested: str) -> torch.device:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "auto"
        else torch.device(requested)
    )
    if device.type == "cuda":
        # Every window has the same shape, so letting cuDNN pick its algorithms once
        # pays for itself many times over across five folds.
        torch.backends.cudnn.benchmark = True
    return device


class _BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class MelResNet(nn.Module):
    """Stem, four residual stages, global pooling, linear head."""

    def __init__(
        self,
        n_classes: int,
        base_width: int = 32,
        n_stages: int = 4,
        blocks_per_stage: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        in_channels = base_width
        for stage in range(n_stages):
            out_channels = base_width * (2**stage)
            for block in range(blocks_per_stage):
                stride = 2 if (block == 0 and stage > 0) else 1
                stages.append(_BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, n_classes)

    @property
    def final_stage(self) -> nn.Module:
        """The block Grad-CAM hooks into."""
        return self.stages[-1]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(pooled))


class SpectrogramAugment:
    """Masking and shifting applied in mel space.

    Gain jitter is absent by design. Windows are normalised individually during
    extraction, so scaling one would be undone before the model ever sees it.
    Pitch shifting is absent because it moves the frequency content the label
    depends on.
    """

    def __init__(self, settings: dict[str, Any]):
        self.enabled = bool(settings.get("enabled", True))
        self.probability = float(settings.get("probability", 0.5))
        self.max_time_shift = float(settings.get("max_time_shift", 0.2))
        self.noise_std = float(settings.get("noise_std", 0.1))
        self.freq_mask_bins = int(settings.get("freq_mask_bins", 8))
        self.time_mask_frames = int(settings.get("time_mask_frames", 32))

    def __call__(self, batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if not self.enabled:
            return batch
        n, _, n_mels, n_frames = batch.shape
        device = batch.device

        def coin() -> torch.Tensor:
            return torch.rand(n, device=device, generator=generator) < self.probability

        if self.max_time_shift > 0:
            span = max(int(self.max_time_shift * n_frames), 1)
            shifts = torch.randint(-span, span + 1, (n,), device=device, generator=generator)
            shifts = torch.where(coin(), shifts, torch.zeros_like(shifts))
            for i, shift in enumerate(shifts.tolist()):
                if shift:
                    batch[i] = torch.roll(batch[i], shifts=shift, dims=-1)

        if self.noise_std > 0:
            noise = torch.randn(batch.shape, device=device, generator=generator) * self.noise_std
            batch = batch + noise * coin().view(n, 1, 1, 1)

        batch = self._mask(batch, coin(), self.freq_mask_bins, n_mels, 2, generator)
        batch = self._mask(batch, coin(), self.time_mask_frames, n_frames, 3, generator)
        return batch

    @staticmethod
    def _mask(
        batch: torch.Tensor,
        selected: torch.Tensor,
        max_width: int,
        axis_size: int,
        axis: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if max_width <= 0:
            return batch
        width = int(min(max_width, axis_size - 1))
        positions = torch.arange(axis_size, device=batch.device)
        for i in torch.nonzero(selected).flatten().tolist():
            size = int(torch.randint(1, width + 1, (1,), generator=generator, device=batch.device))
            start = int(
                torch.randint(
                    0, axis_size - size + 1, (1,), generator=generator, device=batch.device
                )
            )
            span = (positions >= start) & (positions < start + size)
            shape = [1, 1, 1, 1]
            shape[axis] = axis_size
            batch[i] = batch[i].masked_fill(span.view(shape[1:]), 0.0)
        return batch


def _batches(
    view: RowView, labels: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    order = rng.permutation(len(view)) if shuffle else np.arange(len(view))
    for start in range(0, len(order), batch_size):
        positions = order[start : start + batch_size]
        yield view.take(positions), labels[positions]


class SpectrogramCNN(WindowClassifier):
    def __init__(self, model_settings: dict[str, Any], train_settings: dict[str, Any],
                 augment_settings: dict[str, Any]):
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

    def save(self, path: Path) -> None:
        if self.module is None:
            raise RuntimeError("fit must be called before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.module.state_dict(), "settings": self._model_settings},
            path.with_suffix(".pt"),
        )

    @classmethod
    def load(
        cls, path: Path, train_settings: dict[str, Any], n_classes: int
    ) -> SpectrogramCNN:
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
