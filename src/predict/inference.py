"""One recording through the whole path a training clip takes.

Same order and same functions as extraction, so a probability here means what a
probability in the report means. What this adds is the refusals, because the report is
handed a manifest that has already dropped the unusable clips and this is handed an
arbitrary file.

Two of them. It will not upsample a file recorded below the band the model was trained
on, because the empty top of the spectrum is a species label to a classifier and a
silent wrong answer to a user. And it will not pad a fraction of a second out to a
window, because the answer would describe the padding.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.audio.decode import windows_of
from src.audio.io import load as load_audio
from src.config import Config
from src.errors import DaturaError
from src.features import registry as features
from src.features.views import RowView
from src.models import registry as models
from src.models.base import WindowClassifier
from src.models.registry import Loader
from src.results import checkpoint_path


class CannotPredict(DaturaError):
    """Raised when a recording cannot honestly be given to the model."""


def probabilities(cfg: Config, path: Path, names: Sequence[str], fold: int) -> np.ndarray:
    """One probability per species for one file, averaged over its windows.

    More than one model averages them, which is measurably better than either alone:
    the trees and the probe together reach 0.842 accuracy against 0.830 and 0.818, and
    their confident mistakes fall to 0.57%. The audio is decoded and windowed once and
    each model is given the representation it wants.
    """
    # Names first, before any audio is decoded. A typo should cost nothing and name
    # the roster, rather than surfacing as a libsndfile error about the wrong file.
    specs = [models.get(name) for name in names]

    windows = _windows(cfg, path)
    votes = []
    for name, spec in zip(names, specs, strict=True):
        if spec.load is None:
            raise CannotPredict(f"{name} saves no checkpoint, so there is nothing to predict with")

        extractor = features.build_extractor(spec.source, cfg)
        matrix = np.asarray(extractor.transform_batch(windows, cfg.audio.target_sample_rate))
        model = _fitted(cfg, name, spec.load, models.load_settings(spec), fold)

        # One file has no cache behind it, so this wraps an array already in memory.
        # Averaged across windows first, which is how every score in the report is
        # computed. Taking the most confident window instead would report the model's
        # best moment rather than its opinion of the recording.
        per_window = model.predict_proba(RowView.over(matrix))
        votes.append(per_window.mean(axis=0))

    # Then averaged across models. Equal weight, because nothing here has been tuned
    # to pick weights and a validation fold would be needed to do it honestly.
    return np.mean(votes, axis=0)


def _windows(cfg: Config, path: Path) -> np.ndarray:
    """Decode a file the model is allowed to hear, and cut it into windows."""
    signal, native_rate = load_audio(path)
    if native_rate < cfg.audio.min_native_sample_rate:
        raise CannotPredict(
            f"{path.name} is recorded at {native_rate} Hz and the model was trained on "
            f"audio of at least {cfg.audio.min_native_sample_rate} Hz. Upsampling it would "
            "add an empty high band, which a classifier reads as a species."
        )

    seconds = len(signal) / native_rate
    if seconds < cfg.audio.min_clip_seconds:
        raise CannotPredict(
            f"{path.name} is {seconds:.2f}s and the model was trained on clips of at least "
            f"{cfg.audio.min_clip_seconds}s. Padding it out to a window would reflect a "
            "fraction of a second into two, and the answer would describe the padding."
        )

    windows = windows_of(cfg, signal, native_rate)
    if not len(windows):
        raise CannotPredict(f"{path.name} is shorter than one {cfg.audio.window_seconds}s window")
    return windows


def _fitted(cfg: Config, name: str, load: Loader, settings: dict, fold: int) -> WindowClassifier:
    """Read one fold's weights back, through the loader that wrote them.

    Each loader knows its own format, torch writing .pt and xgboost .json, so a missing
    file is reported by the one that looked for it. Testing a suffix here would put that
    knowledge in two places, and it did: the check asked for .pt and refused a set of
    trees sitting on disk beside it.
    """
    saved = checkpoint_path(cfg, name, fold, repeat=0)
    try:
        return load(saved, settings, len(cfg.dataset.species))
    except FileNotFoundError as error:
        raise CannotPredict(
            f"{name} has no fitted fold {fold} to predict with ({error}). Train it, or "
            "use the fold this repository ships."
        ) from error
