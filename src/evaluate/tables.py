"""The tables a report is assembled from.

Everything here reads what training wrote and returns a frame. Nothing here writes
a file, formats markdown or draws a figure, so the same tables serve the report,
the notebooks and anyone poking at results in a shell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.data.manifest import load_manifest
from src.errors import DaturaError
from src.evaluate import metrics
from src.models import registry as models
from src.results import has_results, model_directory, predictions_path


class MissingResults(DaturaError):
    """Raised when a report is asked for before anything has been trained."""


def available_models(cfg: Config) -> list[str]:
    """Models with results on disk, in the order the registry declares them."""
    return [name for name in models.names() if has_results(cfg, name)]


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


def margin_over_control(table: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Each audio model's distance from the floor.

    The control sees no audio, so its score is what recording metadata alone
    achieves. An audio result is only evidence about whales to the extent it clears
    that number.
    """
    control_name = models.control().name
    control = table[table["model"] == control_name]
    if control.empty:
        raise MissingResults(
            f"no {control_name} results found; rerun python -m src.train.xgb without --skip-control"
        )
    floor = float(control[f"{metric}_mean"].iloc[0])

    frame = table[table["model"] != control_name][
        ["model", f"{metric}_mean", f"{metric}_std"]
    ].copy()
    frame.columns = ["model", "mean", "std"]
    frame["control"] = floor
    frame["margin"] = frame["mean"] - floor
    return frame.sort_values("margin", ascending=False)


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
    columns = metrics.probability_columns(len(class_names))

    rows = []
    for name in model_names:
        predictions = pd.read_parquet(predictions_path(cfg, name))
        predictions["shared_rate"] = predictions["clip_id"].map(rate_of_clip).isin(shared_rates)
        for shared, subset in predictions.groupby("shared_rate"):
            per_fold = [
                metrics.score(
                    fold_subset["label"].to_numpy(),
                    fold_subset[columns].to_numpy(),
                    class_names,
                )["macro_f1"]
                for _, fold_subset in subset.groupby("fold")
            ]
            rows.append(
                {
                    "model": name,
                    "subset": "rate shared by species" if shared else "rate unique to a species",
                    "clips": len(subset),
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
