"""Where a network runs, how repeatable that makes it, and how a batch is drawn.

Device choice and backend flags are settings about the machine rather than about
the model, so they live apart from the architecture and the training loop.

Batching and the learning rate schedule sit here for a different reason: two
classifiers in this project need them and neither owns them. The spectrogram network
and the probe over encoder embeddings differ in what they put a batch through, not in
how a batch is cut or how the rate decays.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator

import numpy as np
import torch

from src.features.views import RowView


def batches(
    view: RowView, labels: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Rows and their labels, in blocks, without materialising the whole fold."""
    order = rng.permutation(len(view)) if shuffle else np.arange(len(view))
    for start in range(0, len(order), batch_size):
        positions = order[start : start + batch_size]
        yield view.take(positions), labels[positions]


def schedule(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base rate."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def configure_backend(device: torch.device, deterministic: bool) -> None:
    """Trade throughput for identical runs, or the other way round.

    Left off, cuDNN benchmarks its algorithms once for the fixed window shape and
    keeps the fast one, which pays for itself across five folds but picks whichever
    kernel is quickest on the day. Turned on, the same seed reproduces the same
    weights on the same hardware, at roughly a third less throughput.

    Deterministic CUDA matmul also wants CUBLAS_WORKSPACE_CONFIG=:4096:8 in the
    environment before the process starts. Without it the run still works, it just
    falls back on the operations that cannot be made deterministic.
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def breathe() -> None:
    """Pause after an epoch, when the machine has been asked to run cooler.

    A laptop GPU has no power limit this project can set, so the only way to lower
    sustained draw is to stop asking for it. ``DATURA_EPOCH_COOLDOWN`` is seconds of
    idle after each epoch, which trades wall clock for temperature and changes no
    number: the pause sits between optimiser steps, touches no state, and the seeds,
    the batches and the schedule are all unaffected.
    """
    pause = float(os.environ.get("DATURA_EPOCH_COOLDOWN", "0"))
    if pause > 0:
        time.sleep(pause)
