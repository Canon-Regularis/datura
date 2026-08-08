"""Where the network runs, and how repeatable that makes it.

Device choice and backend flags are settings about the machine, not about the
model, so they live apart from the architecture and the training loop.
"""

from __future__ import annotations

import torch


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
