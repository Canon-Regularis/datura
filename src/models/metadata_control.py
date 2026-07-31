"""The control model, and the reason the rest of the numbers can be read at all.

It sees native sample rate, recording year, clip duration and file size. No audio.

The Watkins collection was recorded across five decades on whatever equipment the
work required, and the equipment tracks the species: every fin whale tape sampled
runs at 600 Hz while sperm whale tapes reach 166 kHz. Any classifier handed raw
recordings can separate species on that basis alone. This model measures how far
that gets you, so the audio results are reported as a margin over it rather than as
a bare accuracy figure.
"""

from __future__ import annotations

from typing import Any

from src.models.gbdt import GradientBoostedTrees

_CONTROL_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 40,
    "tree_method": "hist",
    "n_jobs": -1,
}


def build_metadata_control(seed: int) -> GradientBoostedTrees:
    """A deliberately small model. Its job is to measure a floor. Clearing that floor
    is what the audio models are for."""
    return GradientBoostedTrees({**_CONTROL_PARAMS, "random_state": seed}, name="metadata")
