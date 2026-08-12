"""The training loop the two torch models share.

It was written twice. The probe and the CNN differed on 34 of about 87 lines, and the
34 were the interesting ones: what module to build, how a batch becomes a tensor, and
whether to augment. Everything else, the epoch loop, the optimiser and its schedule,
the validation score, the learning curve, the early stopping and the restore of the
best weights, was the same procedure in two places, and ``predict_proba`` was identical
apart from a default batch size.

Two copies of a training loop do not stay in step. A change to early stopping made in
one of them would leave the other on the old rule, and nothing in the report would say
which model had which, because a learning curve looks equally plausible either way.

Subclasses supply three things and inherit the rest:

``_build``      the module, once the number of classes is known
``_to_tensor``  a batch of features as the tensor that module expects
``_augment``    optional, and the default returns the batch untouched

The order of RNG consumption is fixed here on purpose. ``manual_seed`` runs first, then
the augmentation generator is made, then ``_prepare`` and ``_build``. A generator is
independent of the global stream so making it consumes nothing, and module construction
is the first thing that draws. Both models produced bit identical predictions before and
after this was extracted, on the fold each was checked against.
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.features.views import RowView
from src.models.base import Batch, FoldContext, WindowClassifier, balanced_class_weights
from src.models.runtime import batches, breathe, configure_backend, resolve_device, schedule


class TorchWindowClassifier(WindowClassifier):
    """A window classifier fitted by gradient descent, with early stopping."""

    # Mixed precision is worth having on a GPU and is a source of difference on a CPU,
    # so it is opt in per model rather than inferred. The probe is a linear map over
    # cached vectors and gains nothing from it.
    USES_AMP = False

    def __init__(self, train_settings: dict[str, Any]):
        self._train_settings = dict(train_settings)
        self.device = resolve_device(str(train_settings["device"]))
        self.module: nn.Module | None = None
        self.history: list[dict[str, float]] = []

    @abstractmethod
    def _build(self, n_classes: int) -> nn.Module:
        """The module to fit, moved to this model's device."""

    @abstractmethod
    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        """One batch of features as the tensor ``_build``'s module expects."""

    def _prepare(self, train: Batch) -> None:
        """Anything a model must measure from its training rows before it can be built.

        The probe standardises here, because the width of its head is the width of the
        vectors it is about to see. Most models need nothing.
        """

    def _augment(self, inputs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """A batch on its way into the module. The default changes nothing."""
        return inputs

    def _setting(self, name: str) -> Any:
        """One training setting, which the file must supply.

        No default. Every one of these used to carry one, and the defaults disagreed
        with the file they backed onto, so a misspelled key trained a third longer than
        the config said and printed nothing. The loader validates the file instead.
        """
        return self._train_settings[name]

    def fit(self, train: Batch, validation: Batch, n_classes: int) -> None:
        seed = int(self._setting("seed"))
        configure_backend(self.device, bool(self._setting("deterministic")))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        self._prepare(train)
        self.module = self._build(n_classes)

        weights = balanced_class_weights(train.labels, n_classes)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

        epochs = int(self._setting("epochs"))
        batch_size = int(self._setting("batch_size"))
        warmup = int(self._setting("warmup_epochs"))
        patience = int(self._setting("early_stopping_patience"))
        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=float(self._setting("lr")),
            weight_decay=float(self._setting("weight_decay")),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: schedule(epoch, warmup, epochs)
        )
        use_amp = self.USES_AMP and self.device.type == "cuda"
        scaler = torch.amp.GradScaler(self.device.type, enabled=use_amp)

        best_score = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0

        for epoch in range(epochs):
            self.module.train()
            total_loss = 0.0
            seen = 0
            for features, labels in batches(train.features, train.labels, batch_size, True, rng):
                inputs = self._augment(self._to_tensor(features), generator)
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

            # A laptop GPU trains this for hours, so the runner is given a moment to
            # shed heat between epochs rather than between folds alone.
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

        # Twice the training batch, because inference holds no gradients and the same
        # memory takes twice as many rows.
        batch_size = int(self._setting("batch_size")) * 2
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
