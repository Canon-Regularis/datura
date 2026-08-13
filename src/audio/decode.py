"""One recording, turned into the windows a model was trained on.

Three steps that have to agree everywhere: decode, resample to the target rate, cut
into overlapping windows. Cache extraction did them and so did the prediction command,
each reading the five geometry settings out of the configuration itself, so a change
to the padding mode or the window cap had two places to reach and nothing checking it
reached both. A prediction whose windows differ from the training windows is wrong in
a way that produces an ordinary looking probability.

The two callers differ in what they refuse rather than in what they do. Extraction
works off a manifest that has already dropped the unusable clips, so anything reaching
it is known good. The prediction command is handed an arbitrary file and has to decide
whether the model may honestly be shown it, which is why it decodes first and windows
after its own guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.audio.io import load as load_audio
from src.audio.resample import to_target_rate
from src.audio.windows import split_into_windows
from src.config import Config


def windows_of(cfg: Config, signal: np.ndarray, native_rate: int) -> np.ndarray:
    """Resample a decoded signal and cut it into windows, on this corpus' geometry."""
    resampled = to_target_rate(signal, native_rate, cfg.audio.target_sample_rate)
    return split_into_windows(
        resampled,
        cfg.audio.window_samples,
        cfg.audio.hop_samples,
        cfg.audio.pad_mode,
        cfg.audio.max_windows_per_clip,
    )


def decode_to_windows(cfg: Config, path: Path) -> np.ndarray:
    """Read a file off disk and give back its windows, for a caller with nothing to check."""
    signal, native_rate = load_audio(path)
    return windows_of(cfg, signal, native_rate)
