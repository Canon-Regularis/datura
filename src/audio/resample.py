"""Rate conversion onto the common analysis band.

Upsampling is rejected rather than performed. A file recorded at 5120 Hz and
upsampled to 10 kHz has nothing above 2560 Hz, and a classifier will happily use
that empty band as a species label instead of learning anything about the animal.
Dropping those files costs data but keeps the result meaningful.
"""

from __future__ import annotations

import numpy as np
import soxr


class UpsamplingRejected(ValueError):
    """Raised when a signal's native rate is below the common target rate."""

    def __init__(self, native_rate: int, target_rate: int) -> None:
        super().__init__(
            f"native rate {native_rate} Hz is below the target rate {target_rate} Hz; "
            "this file must be excluded, not upsampled"
        )
        self.native_rate = native_rate
        self.target_rate = target_rate


def to_target_rate(signal: np.ndarray, native_rate: int, target_rate: int) -> np.ndarray:
    """Resample down to ``target_rate``, band-limiting as it goes."""
    if native_rate < target_rate:
        raise UpsamplingRejected(native_rate, target_rate)
    if native_rate == target_rate:
        return np.ascontiguousarray(signal, dtype=np.float32)
    converted = soxr.resample(signal.astype(np.float32), native_rate, target_rate, quality="HQ")
    return np.ascontiguousarray(converted, dtype=np.float32)
