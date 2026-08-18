"""Compare representations without choosing one on the number being published.

A spectrogram setting is a hyperparameter, and picking it by looking at the test folds
is the same mistake as picking a threshold there. The difference is that a threshold
looks obviously wrong and a spectrogram setting looks like engineering.

So this ranks on ``fold_metrics_validation.csv``, which is scored on the rows every
fold held out for early stopping. There is no argument for ranking on test that would
survive being written down, so the command does not offer it.

Ranking alone is not enough, because a table always has a leader. The folds of a
repeated grouped split share most of their training rows, and once that overlap is
corrected for, this design separates paired differences of roughly 0.05 macro-F1 and
nothing finer. A leader inside that band is a coin flip, so the margin is tested before
the winner is measured on test at all.

The settings live under ``configs/sweep/``. ``experiment_configs`` globs ``configs/*.yaml``
and is not recursive, so a setting cannot wander into the multiplicity correction or the
reproduce job while it is still a candidate.

Usage:
    python -m src.evaluate.sweep [--model xgboost_centred] [--metric macro_f1]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, Config, load_config
from src.errors import DaturaError
from src.logging_config import configure
from src.results import clip_metrics_path, validation_metrics_path
from src.uncertainty import fold_scores, paired_difference, shared_folds

logger = logging.getLogger(__name__)

SWEEP_DIRECTORY = PROJECT_ROOT / "configs" / "sweep"
BASELINE = PROJECT_ROOT / "configs" / "base.yaml"


class NothingToCompare(DaturaError):
    """Raised when no candidate has been fitted yet."""


def candidates() -> list[Path]:
    """The baseline first, then every sweep setting, in a stable order."""
    return [BASELINE, *sorted(SWEEP_DIRECTORY.glob("*.yaml"))]


def _mean(path: Path, metric: str) -> tuple[float, float, int] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if metric not in frame.columns:
        return None
    return float(frame[metric].mean()), float(frame[metric].std(ddof=1)), len(frame)


def _worst_class(path: Path) -> tuple[str, float] | None:
    """The lowest scoring class and its F1, which is what a macro average hides.

    Read generically from the ``f1_<class>`` columns rather than by naming a species,
    so this says nothing about which archive it is looking at. A macro average over
    three classes moves when the weakest one moves, and reporting the mean alone would
    let a setting that helped two classes and hurt the third look like an improvement.
    """
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    columns = [c for c in frame.columns if c.startswith("f1_")]
    if not columns:
        return None
    means = {c.removeprefix("f1_"): float(frame[c].mean()) for c in columns}
    name = min(means, key=lambda key: means[key])
    return name, means[name]


def _row(cfg: Config, model: str, metric: str) -> dict | None:
    path = validation_metrics_path(cfg, model)
    validation = _mean(path, metric)
    if validation is None:
        return None
    mean, spread, folds = validation
    worst = _worst_class(path)
    return {
        "setting": cfg.name,
        "n_fft": cfg.spectrogram.n_fft,
        "n_mels": cfg.spectrogram.n_mels,
        "window_seconds": cfg.audio.window_seconds,
        f"validation_{metric}": mean,
        "validation_std": spread,
        "worst_class": "" if worst is None else worst[0],
        "worst_class_f1": float("nan") if worst is None else worst[1],
        "folds": folds,
    }


def compare(model: str = "xgboost_centred", metric: str = "macro_f1") -> pd.DataFrame:
    """Every fitted candidate, ranked by how it scored on the validation rows."""
    rows = [row for path in candidates() if (row := _row(load_config(path), model, metric))]
    if not rows:
        raise NothingToCompare(
            f"no candidate has {validation_metrics_path(load_config(BASELINE), model).name}; "
            f"fit one with python -m src.train.xgb --config configs/sweep/<setting>.yaml "
            f"--only {model}"
        )
    table = pd.DataFrame(rows).sort_values(f"validation_{metric}", ascending=False)
    return table.reset_index(drop=True)


def margin_over_baseline(
    candidate: Config, model: str, metric: str
) -> tuple[float, float, float, float, int] | None:
    """The winner against the baseline on validation, corrected for fold overlap.

    Ranking picks a leader out of any table, including one where every candidate is
    the same setting measured twice. What decides whether the leader is worth adopting
    is whether its margin is larger than what these folds can resolve, and the folds
    of a repeated split share most of their training rows, so a plain paired t test
    reads far too confident. ``paired_difference`` applies the Nadeau and Bengio
    correction the rest of the project already uses for every published comparison.

    Measured on validation, because test is read once and only after the choice.
    """
    left = validation_metrics_path(candidate, model)
    right = validation_metrics_path(load_config(BASELINE), model)
    if not (left.exists() and right.exists()) or left == right:
        return None
    a, b = shared_folds(
        fold_scores(pd.read_csv(left), metric), fold_scores(pd.read_csv(right), metric)
    )
    if a.empty:
        return None
    result = paired_difference(a, b)
    return result.difference, result.low, result.high, result.p_value, result.n_folds


def report(model: str = "xgboost_centred", metric: str = "macro_f1") -> pd.DataFrame:
    """Rank on validation, test whether the leader is separable, then read test once."""
    table = compare(model, metric)
    logger.info(
        "\nranked on validation, which is what the choice is allowed to see\n%s",
        table.round(4).to_string(index=False),
    )

    winner = table.iloc[0]["setting"]
    cfg = load_config(next(c for c in candidates() if load_config(c).name == winner))
    logger.info("\nleader on validation: %s", winner)

    margin = margin_over_baseline(cfg, model, metric)
    if margin is None:
        logger.info("  the leader is the baseline, so there is nothing to adopt")
        return table

    difference, low, high, p_value, folds = margin
    logger.info(
        "  margin over the baseline on validation: %+.4f [%+.4f, %+.4f], p = %.3f, %d folds",
        difference,
        low,
        high,
        p_value,
        folds,
    )
    if p_value < 0.05:
        logger.info("  separable at p < 0.05, so it is worth measuring on test")
        measured = _mean(clip_metrics_path(cfg, model), metric)
        if measured is not None:
            logger.info("  test %s %.4f +/- %.4f over %d folds", metric, *measured)
    else:
        logger.info(
            "  these folds resolve differences of about %.3f, and this one is smaller, "
            "so the baseline stands and test stays unread",
            (high - low) / 2,
        )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--model", default="xgboost_centred", help="which model to compare across settings"
    )
    parser.add_argument("--metric", default="macro_f1", help="which metric to rank on")
    args = parser.parse_args(argv)

    configure(verbose=False, quiet=False)
    report(args.model, args.metric)
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except DaturaError as error:
        logger.error("%s", error)
        sys.exit(1)
