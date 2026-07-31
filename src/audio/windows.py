"""Cutting a clip into fixed-length analysis windows.

Short clips are reflect-padded rather than zero-padded. A block of digital silence
at the end of a clip is a length cue, and clip length differs systematically
between species, so zero padding would hand the model a shortcut.
"""

from __future__ import annotations

import numpy as np


def pad_to_length(signal: np.ndarray, length: int, mode: str = "reflect") -> np.ndarray:
    """Extend a signal to ``length`` samples without introducing silence."""
    if signal.size >= length:
        return signal
    deficit = length - signal.size
    # Reflection needs at least two samples to bounce between.
    effective_mode = mode if signal.size > 1 else "constant"
    return np.pad(signal, (0, deficit), mode=effective_mode)


def window_starts(n_samples: int, window: int, hop: int, max_windows: int = 0) -> list[int]:
    """Start offsets covering the whole signal, including a tail-aligned final window.

    Without the final window the last fraction of a clip shorter than one hop would
    never be seen by the model.

    ``max_windows`` thins long clips down to that many evenly spaced windows. Clip
    length is wildly uneven in this collection, from under a second to twenty-four
    minutes, so without a cap a handful of the longest recordings would supply most
    of the training set and the model would mostly be learning those tapes.
    """
    if window <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    if n_samples <= window:
        return [0]

    starts = list(range(0, n_samples - window + 1, hop))
    if starts[-1] + window < n_samples:
        starts.append(n_samples - window)

    if max_windows and len(starts) > max_windows:
        picks = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(picks.tolist())]
    return starts


def split_into_windows(
    signal: np.ndarray,
    window: int,
    hop: int,
    pad_mode: str = "reflect",
    max_windows: int = 0,
) -> np.ndarray:
    """Return a ``(n_windows, window)`` float32 array covering the signal."""
    padded = pad_to_length(signal, window, mode=pad_mode)
    starts = window_starts(padded.size, window, hop, max_windows)
    stacked = np.stack([padded[s : s + window] for s in starts])
    return np.ascontiguousarray(stacked, dtype=np.float32)
