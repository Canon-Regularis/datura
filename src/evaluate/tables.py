"""The tables a report is assembled from.

Everything here reads what training wrote and returns a frame. Nothing here writes
a file, formats markdown or draws a figure, so the same tables serve the report,
the notebooks and anyone poking at results in a shell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import scoring, uncertainty
from src.config import Config
from src.data.manifest import load_manifest
from src.errors import DaturaError
from src.evaluate import families
from src.results import model_directory, predictions_path


class MissingResults(DaturaError):
    """Raised when a report is asked for before anything has been trained."""


def comparison(cfg: Config, model_names: list[str]) -> pd.DataFrame:
    """One row per model, one column per metric mean and spread."""
    rows = []
    for name in model_names:
        summary = pd.read_csv(model_directory(cfg, name) / "summary.csv")
        record: dict[str, object] = {"model": name}
        for _, entry in summary.iterrows():
            record[f"{entry['metric']}_mean"] = entry["mean"]
            record[f"{entry['metric']}_std"] = entry["std"]
        rows.append(record)
    return pd.DataFrame(rows)


def headline(table: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """One metric per model, with its spread across folds."""
    frame = table[["model", f"{metric}_mean", f"{metric}_std"]].copy()
    frame.columns = ["model", "mean", "std"]
    return frame


def _fold_scores(cfg: Config, name: str, metric: str) -> pd.Series:
    return uncertainty.fold_scores(
        pd.read_csv(model_directory(cfg, name) / "fold_metrics_clip.csv"), metric
    )


def family_margins(cfg: Config, family: families.Family, metric: str = "macro_f1") -> pd.DataFrame:
    """Every model in one family against its control, with what the design resolves.

    The margin alone invites more confidence than a handful of recordings supports.
    The three columns beside it say how much: the interval the paired folds allow,
    the p value, and how many of those folds pointed the same way. That last one is
    worth its space, because a direction holding in most folds is informative even
    where the p value settles nothing.
    """
    control = _fold_scores(cfg, family.control, metric)

    rows = []
    for name in family.members:
        left, right = uncertainty.shared_folds(_fold_scores(cfg, name, metric), control)
        difference = uncertainty.paired_difference(left, right)
        rows.append(
            {
                "model": name,
                "mean": float(left.mean()),
                "control": float(right.mean()),
                "margin": difference.difference,
                "low": difference.low,
                "high": difference.high,
                "p_value": difference.p_value,
                "agreeing": difference.folds_agreeing,
                "folds": difference.n_folds,
            }
        )
    return pd.DataFrame(rows).sort_values("margin", ascending=False)


def family_intervals(
    cfg: Config,
    family: families.Family,
    metric: str = "macro_f1",
    resamples: int = uncertainty.DEFAULT_RESAMPLES,
) -> pd.DataFrame:
    """Each model's own score with the range the recordings support.

    The interval comes from resampling tapes rather than clips, so it answers what
    a different draw of recordings would have produced. That is a wider and more
    honest question than the spread across folds.
    """
    rows = []
    for name in family.names:
        predictions = pd.read_parquet(predictions_path(cfg, name))
        if "repeat" in predictions.columns:
            # One prediction per clip. A repeated run holds a full pass per repeat,
            # and pooling them would score some clips ten times over.
            predictions = predictions[predictions["repeat"] == 0].reset_index(drop=True)
        interval = uncertainty.bootstrap_metric(
            predictions, list(family.class_names), metric=metric, resamples=resamples
        )
        rows.append(
            {
                "model": name,
                "estimate": interval.estimate,
                "low": interval.low,
                "high": interval.high,
                "tapes": predictions[uncertainty.GROUP_COLUMN].nunique(),
            }
        )
    return pd.DataFrame(rows)


def ambiguity(cfg: Config, model_names: list[str]) -> pd.DataFrame:
    """Each model scored with and without the equipment giveaway.

    A native sample rate used by one species hands the answer to anything that can
    see it; a rate shared by several does not. Splitting the test clips that way
    asks the question the headline number cannot: when the recording is not a
    giveaway, does listening to it still help?
    """
    manifest = load_manifest(cfg, kept_only=True)
    species_per_rate = manifest.groupby("native_sample_rate")["species"].nunique()
    shared_rates = set(species_per_rate[species_per_rate > 1].index)
    rate_of_clip = manifest.set_index("clip_id")["native_sample_rate"]

    class_names = list(cfg.dataset.species)
    columns = scoring.probability_columns(len(class_names))

    rows = []
    for name in model_names:
        predictions = pd.read_parquet(predictions_path(cfg, name))
        predictions["shared_rate"] = predictions["clip_id"].map(rate_of_clip).isin(shared_rates)

        # One score per split, and a repeated run has a split per repeat per fold.
        # Grouping on the fold alone would pool ten different splits into each of five
        # rows and report the spread across folds as if it were the spread across runs.
        split = [key for key in ("repeat", "fold") if key in predictions.columns]

        for shared, subset in predictions.groupby("shared_rate"):
            per_fold = [
                scoring.score(
                    fold_subset["label"].to_numpy(),
                    fold_subset[columns].to_numpy(),
                    class_names,
                )["macro_f1"]
                for _, fold_subset in subset.groupby(split)
            ]
            rows.append(
                {
                    "model": name,
                    "subset": "rate shared by species" if shared else "rate unique to a species",
                    # Distinct clips, not prediction rows. A repeated run holds one row per
                    # clip per repeat, so counting rows would put a ten repeat model and a
                    # single split model on different scales in the same column.
                    "clips": subset["clip_id"].nunique(),
                    "macro_f1_mean": float(np.mean(per_fold)),
                    "macro_f1_std": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def per_species_recall(table: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Mean recall per species per model, named by species rather than by column."""
    frame = table[["model", *[f"recall_{name}_mean" for name in class_names]]].copy()
    frame.columns = ["model", *class_names]
    return frame
