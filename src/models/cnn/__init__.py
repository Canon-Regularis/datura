"""The log mel network.

Split by responsibility: ``network`` is the architecture, ``augment`` is what
happens to a batch on the way in, ``runtime`` is the device and its determinism
flags, and ``classifier`` is the fitting procedure that uses all three.

Callers import from this package, so a part can move without breaking them.
"""

from __future__ import annotations

from src.models.cnn.augment import SpectrogramAugment
from src.models.cnn.classifier import SpectrogramCNN
from src.models.cnn.network import MelResNet
from src.models.cnn.runtime import configure_backend, resolve_device

__all__ = [
    "MelResNet",
    "SpectrogramAugment",
    "SpectrogramCNN",
    "configure_backend",
    "resolve_device",
]
