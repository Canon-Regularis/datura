"""Name the species in one recording, or decline to.

Everything else in this project scores a model against held out folds. This points
one at a file and says what it thinks, which is the thing a person actually wants
from it, and it is deliberately allowed to refuse.

Refusing matters, and it matters less than picking a model whose confidence means
something. Forced to answer for every clip the trees are right 81% of the time, and
allowed to decline the least confident third they are right 90% of the time. The cut
off is read from the model's own coverage curve rather than chosen, because the same
probability means different things to different models: a 0.6 threshold declines 31%
of the trees' answers and 5% of the network's.

Abstention filters uncertainty, not error. It removes the ambiguous cases and leaves
the confident mistakes, so a model that is often wrong while sure stays wrong while
sure. That is why the model shipped here is the one that is confidently wrong least
often rather than the one that scores best on any single fold.

Two things it will not do. It will not upsample a file recorded below the band the
model was trained on, because the empty top of the spectrum is a species label to a
classifier and a silent wrong answer to a user. And it will not pretend a low
confidence prediction is a prediction.

Usage:
    python -m src.predict recording.wav [--model cnn_small] [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio.io import load as load_audio
from src.audio.resample import to_target_rate
from src.audio.windows import split_into_windows
from src.config import Config, load_config
from src.errors import DaturaError
from src.evaluate import coverage
from src.features import registry as features
from src.features.views import RowView
from src.models import registry as models
from src.results import checkpoint_path, coverage_path

logger = logging.getLogger(__name__)

# The trees rather than the network. Pooled over all 41,600 held out predictions the
# network is wrong while above 90% confident on 8.62% of them, and the trees on 0.10%.
# Eighty five times fewer confident mistakes, and a higher score besides. Shipping the
# network was a mistake made because it was the only model that saved itself.
DEFAULT_MODEL = "xgboost"
DEFAULT_FOLD = 0

# What accuracy a prediction has to be worth before it is offered as one. The cut off
# probability is then read from the model's own coverage curve, because a fixed one is
# meaningless across models.
#
# This was a hardcoded 0.6 and that was a bug. Measured on held out predictions, a 0.6
# cut declines 31.5% of XGBoost's answers and only 5.3% of cnn_small's, because the
# network is far more overconfident: to reach the same 90% accuracy cnn_small needs
# 0.954 where XGBoost needs 0.591. One number cannot serve both, and the one chosen
# left the shipped model calling a 99.7% wrong answer HIGH confidence.
TARGET_ACCURACY = 0.90


class CannotPredict(DaturaError):
    """Raised when a recording cannot honestly be given to the model."""


def probabilities(cfg: Config, path: Path, names: Sequence[str], fold: int) -> np.ndarray:
    """One probability per species for one file, averaged over its windows.

    The whole path a training clip takes, in the same order and through the same
    functions, so a prediction here means what a prediction in the report means.

    More than one model averages them, which is measurably better than either alone:
    the trees and the probe together reach 0.842 accuracy against 0.830 and 0.818, and
    their confident mistakes fall to 0.57%. The audio is decoded and windowed once and
    each model is given the representation it wants.
    """
    # Names first, before any audio is decoded. A typo should cost nothing and name
    # the roster, rather than surfacing as a libsndfile error about the wrong file.
    specs = [models.get(name) for name in names]

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

    resampled = to_target_rate(signal, native_rate, cfg.audio.target_sample_rate)
    windows = split_into_windows(
        resampled,
        cfg.audio.window_samples,
        cfg.audio.hop_samples,
        cfg.audio.pad_mode,
        cfg.audio.max_windows_per_clip,
    )
    if not len(windows):
        raise CannotPredict(f"{path.name} is shorter than one {cfg.audio.window_seconds}s window")

    votes = []
    for name, spec in zip(names, specs, strict=True):
        if spec.load is None:
            raise CannotPredict(f"{name} saves no checkpoint, so there is nothing to predict with")

        extractor = features.build_extractor(spec.source, cfg)
        matrix = np.asarray(extractor.transform_batch(windows, cfg.audio.target_sample_rate))

        # Each loader knows its own format, torch writing .pt and xgboost .json, so a
        # missing file is reported by the one that looked for it. Testing a suffix here
        # would put that knowledge in two places, and it did: the check asked for .pt
        # and refused a set of trees sitting on disk beside it.
        saved = checkpoint_path(cfg, name, fold, repeat=0)
        try:
            model = spec.load(saved, models.load_settings(spec), len(cfg.dataset.species))
        except FileNotFoundError as error:
            raise CannotPredict(
                f"{name} has no fitted fold {fold} to predict with ({error}). Train it, or "
                "use the fold this repository ships."
            ) from error

        # One file has no cache behind it, so this wraps an array already in memory.
        # Averaged across windows first, which is how every score in the report is
        # computed. Taking the most confident window instead would report the model's
        # best moment rather than its opinion of the recording.
        per_window = model.predict_proba(RowView.over(matrix))
        votes.append(per_window.mean(axis=0))

    # Then averaged across models. Equal weight, because nothing here has been tuned
    # to pick weights and a validation fold would be needed to do it honestly.
    return np.mean(votes, axis=0)


def curve_for(cfg: Config, name: str) -> pd.DataFrame | None:
    """This model's measured accuracy against coverage, or nothing if it has none."""
    path = coverage_path(cfg)
    if not path.exists():
        return None
    table = pd.read_csv(path)
    rows = table[table["model"] == name]
    return rows if len(rows) else None


@dataclass(frozen=True)
class Standing:
    """Whether a model may answer at all, and what its answer is worth.

    Three states, and collapsing two of them is how this command came to print
    ``Confidence : HIGH`` beside a model that is right 49.5% of the time. There was one
    nullable float, and ``None`` meant both "this configuration has no curve" and "this
    model never reaches the target at any coverage". The second is a refusal and the
    first is an apology, and the code took the confident branch for both.
    """

    cut_off: float | None
    ceiling: float | None
    matched: pd.Series | None

    @property
    def has_curve(self) -> bool:
        """Whether anything is known about how often this model is right."""
        return self.ceiling is not None

    @property
    def reaches_target(self) -> bool:
        """Whether declining the least confident clips ever earns the target accuracy."""
        return self.cut_off is not None


def threshold_for(curve: pd.DataFrame | None, target: float = TARGET_ACCURACY) -> float | None:
    """The probability below which this model should decline, to earn ``target``.

    Taken from the curve rather than chosen, so it adapts to how confident a given
    model happens to be. A network whose probabilities are all above 0.9 needs a far
    higher cut off than a tree whose probabilities spread out, and asking both to
    clear the same number gives one of them an abstention rule that never fires.

    ``None`` means no coverage level reaches the target, which is a model that should
    not be answering. Callers want ``standing``, which can tell that apart from a
    configuration that has no curve at all.
    """
    if curve is None:
        return None
    good = curve[curve["accuracy"] >= target].sort_values("coverage", ascending=False)
    return float(good.iloc[0]["threshold"]) if len(good) else None


def standing(
    cfg: Config, name: str, confidence: float, target: float = TARGET_ACCURACY
) -> Standing:
    """Everything this model's own curve says about a prediction of this confidence."""
    curve = curve_for(cfg, name)
    if curve is None:
        return Standing(cut_off=None, ceiling=None, matched=None)
    return Standing(
        cut_off=threshold_for(curve, target),
        ceiling=float(curve["accuracy"].max()),
        matched=coverage.band(curve, confidence),
    )


def band(cfg: Config, name: str, confidence: float) -> pd.Series | None:
    """What accuracy this confidence earned on held out recordings."""
    curve = curve_for(cfg, name)
    return coverage.band(curve, confidence) if curve is not None else None


def render(cfg: Config, name: str, scores: np.ndarray, verdict: Standing) -> str:
    """The report a person reads."""
    species = list(cfg.dataset.species)
    order = np.argsort(scores)[::-1]
    lines = ["", "Species prediction", "-" * 44]
    lines += [f"  {species[i]:<22s} {scores[i] * 100:5.1f}%" for i in order]
    lines.append("")

    best = int(order[0])
    if verdict.has_curve and not verdict.reaches_target:
        lines += [
            "  Prediction : WITHHELD",
            f"  Confidence : this model never reaches {TARGET_ACCURACY:.0%} accuracy at any",
            f"               coverage. Its best is {verdict.ceiling:.1%}, so no threshold",
            "               makes an answer from it worth acting on.",
        ]
    elif verdict.reaches_target and scores[best] < verdict.cut_off:
        lines += [
            "  Prediction : UNCERTAIN",
            f"  Confidence : LOW, below the {verdict.cut_off:.3f} this model needs to be "
            f"{TARGET_ACCURACY:.0%} accurate",
            "",
            "  This recording should not be classified automatically. The model is not",
            "  separating the species here.",
        ]
    else:
        lines.append(f"  Prediction : {species[best]}")
        matched = verdict.matched
        if matched is not None:
            lines.append(
                f"  Confidence : {_word(matched['coverage'])}, and on held out recordings "
                f"this model was {matched['accuracy']:.1%} accurate"
            )
            lines.append(
                f"               on the most confident {matched['coverage']:.0%} of its predictions"
            )
        elif verdict.has_curve:
            lines.append(
                "  Confidence : below the weakest band this curve measures, so how often it"
            )
            lines.append("               is right at this confidence is unknown")
        else:
            lines.append("  Confidence : no coverage table for this configuration")

    lines += ["", f"  model: {name}, fold {DEFAULT_FOLD}, trained on {cfg.name}", ""]
    return "\n".join(lines)


def _word(coverage_level: float) -> str:
    if coverage_level <= 0.5:
        return "HIGH"
    return "MODERATE" if coverage_level <= 0.8 else "LOW"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path, help="a wav file to identify")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "one model, or several comma separated to average them. "
            f"'{coverage.ENSEMBLE_NAME}' is the best measured combination and needs the "
            "encoder weights; the default runs offline"
        ),
    )
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=TARGET_ACCURACY,
        help="decline any prediction not confident enough to have earned this accuracy",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if not args.recording.exists():
        raise CannotPredict(f"{args.recording} not found")

    chosen = [name.strip() for name in args.model.split(",") if name.strip()]
    for name in chosen:
        models.get(name)  # raises UnknownModel with the roster, before any audio is read

    # A committed curve exists for the pair under the name it is stored by, so an
    # ensemble is measured rather than guessed at, exactly like a single model.
    label = "+".join(chosen)
    scores = probabilities(cfg, args.recording, chosen, args.fold)
    verdict = standing(cfg, label, float(scores.max()), args.target_accuracy)
    print(render(cfg, label, scores, verdict))
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    try:
        sys.exit(main())
    except DaturaError as error:
        print(f"\n  {error}\n", file=sys.stderr)
        sys.exit(1)
