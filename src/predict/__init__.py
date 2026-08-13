"""Name the species in one recording, or decline to.

Everything else in this project scores a model against held out folds. This points one
at a file and says what it thinks, which is the thing a person actually wants from it,
and it is deliberately allowed to refuse.

Refusing matters, and it matters less than picking a model whose confidence means
something. Forced to answer for every clip the trees are right 81% of the time, and
allowed to decline the least confident third they are right 90% of the time.

Three parts, because a command that decodes audio, decides whether to answer and writes
English has three reasons to change. ``inference`` runs a recording through the training
path, ``policy`` reads the model's own coverage curve to decide what an answer is worth,
and ``report`` writes it down. They only meet in ``__main__``.

Usage:
    python -m src.predict recording.wav [--model xgboost] [--config configs/base.yaml]
"""

from __future__ import annotations

from src.predict.inference import CannotPredict, probabilities
from src.predict.policy import (
    DEFAULT_FOLD,
    DEFAULT_MODEL,
    TARGET_ACCURACY,
    Standing,
    band,
    curve_for,
    standing,
    threshold_for,
)
from src.predict.report import render

__all__ = [
    "DEFAULT_FOLD",
    "DEFAULT_MODEL",
    "TARGET_ACCURACY",
    "CannotPredict",
    "Standing",
    "band",
    "curve_for",
    "probabilities",
    "render",
    "standing",
    "threshold_for",
]
