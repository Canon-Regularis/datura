"""Assemble every model's results into one comparison, with figures and a summary.

The summary leads with the metadata control rather than with the best score. An
audio model that beats the control by two points has not shown much about whale
vocalisation, and that has to be visible without reading a table.

Usage:
    python -m src.evaluate.report [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.manifest import load_manifest
from src.evaluate import metrics, plots

MODELS = ("xgboost", "cnn", "cnn_small", "metadata")
CONTROL = "metadata"


class ReportError(RuntimeError):
    pass


def available_models(cfg: Config) -> list[str]:
    root = cfg.paths.reports / cfg.name
    return [name for name in MODELS if (root / name / "summary.csv").exists()]


def comparison_table(cfg: Config, models: list[str]) -> pd.DataFrame:
    """One row per model, one column per metric mean and spread."""
    rows = []
    for name in models:
        summary = pd.read_csv(cfg.paths.reports / cfg.name / name / "summary.csv")
        record: dict[str, object] = {"model": name}
        for _, entry in summary.iterrows():
            record[f"{entry['metric']}_mean"] = entry["mean"]
            record[f"{entry['metric']}_std"] = entry["std"]
        rows.append(record)
    return pd.DataFrame(rows)


def _headline(table: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    frame = table[["model", f"{metric}_mean", f"{metric}_std"]].copy()
    frame.columns = ["model", "mean", "std"]
    return frame


def _margin_over_control(table: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    control = table[table["model"] == CONTROL]
    if control.empty:
        raise ReportError(
            "no metadata control results found; "
            "rerun python -m src.train.xgb without --skip-control"
        )
    floor = float(control[f"{metric}_mean"].iloc[0])
    frame = table[table["model"] != CONTROL][["model", f"{metric}_mean", f"{metric}_std"]].copy()
    frame.columns = ["model", "mean", "std"]
    frame["control"] = floor
    frame["margin"] = frame["mean"] - floor
    return frame.sort_values("margin", ascending=False)


def ambiguity_breakdown(cfg: Config, models: list[str]) -> pd.DataFrame:
    """Score each model separately on clips whose sample rate does and does not
    identify the species on its own.

    A native rate used by only one species hands the answer to any model that can
    see it. A rate shared by several does not. Splitting the test clips that way
    asks the question the headline number cannot: when the recording equipment is
    not a giveaway, does listening to the audio still help?
    """
    manifest = load_manifest(cfg, kept_only=True)
    species_per_rate = manifest.groupby("native_sample_rate")["species"].nunique()
    shared_rates = set(species_per_rate[species_per_rate > 1].index)
    rate_of_clip = manifest.set_index("clip_id")["native_sample_rate"]

    class_names = list(cfg.dataset.species)
    probability_columns = [f"p{i}" for i in range(len(class_names))]

    rows = []
    for name in models:
        predictions = pd.read_parquet(
            cfg.paths.reports / cfg.name / name / "clip_predictions.parquet"
        )
        predictions["shared_rate"] = predictions["clip_id"].map(rate_of_clip).isin(shared_rates)
        for shared, subset in predictions.groupby("shared_rate"):
            per_fold = []
            for _, fold_subset in subset.groupby("fold"):
                scores = metrics.score(
                    fold_subset["label"].to_numpy(),
                    fold_subset[probability_columns].to_numpy(),
                    class_names,
                )
                per_fold.append(scores["macro_f1"])
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


def _write_markdown(cfg: Config, table: pd.DataFrame, margins: pd.DataFrame, directory: Path,
                    figures: list[Path], ambiguity: pd.DataFrame) -> Path:
    class_names = list(cfg.dataset.species)
    lines = [
        f"# Results: {cfg.name}",
        "",
        f"Species: {', '.join(class_names)}  ",
        f"Common band: 0 to {cfg.audio.nyquist:.0f} Hz at {cfg.audio.target_sample_rate} Hz  ",
        f"Folds: {cfg.split.n_folds}, grouped by tape  ",
        f"Windows: {cfg.audio.window_seconds} s, hop {cfg.audio.hop_seconds} s, "
        f"at most {cfg.audio.max_windows_per_clip} per clip",
        "",
        "## Margin over the metadata control",
        "",
        "The control sees native sample rate, year, clip duration and file size, and no audio.",
        "Its score is the floor an audio model has to clear.",
        "",
        margins.round(4).to_markdown(index=False),
        "",
        "## All models",
        "",
        _headline(table).round(4).to_markdown(index=False),
        "",
        "## Per species recall",
        "",
    ]

    recall = table[["model", *[f"recall_{n}_mean" for n in class_names]]].copy()
    recall.columns = ["model", *class_names]
    lines += [recall.round(4).to_markdown(index=False), ""]

    lines += [
        "## With and without the equipment giveaway",
        "",
        "Test clips split by whether their native sample rate is used by one species or",
        "several. On the shared-rate subset the recording cannot identify the species by",
        "itself, so that column is where audio has to earn its result.",
        "",
        ambiguity.round(4).to_markdown(index=False),
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{path.name}`" for path in figures]
    lines += [
        "",
        "Every figure has a CSV of the same name beside it or in the model directory.",
        "",
    ]

    path = directory / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build(cfg: Config) -> Path:
    directory = cfg.paths.reports / cfg.name
    models = available_models(cfg)
    if not models:
        raise ReportError(
            f"no results under {directory}; run python -m src.train.xgb and python -m src.train.cnn"
        )

    class_names = list(cfg.dataset.species)
    table = comparison_table(cfg, models)
    table.to_csv(directory / "comparison.csv", index=False)
    margins = _margin_over_control(table)
    margins.to_csv(directory / "margin_over_control.csv", index=False)
    ambiguity = ambiguity_breakdown(cfg, models)
    ambiguity.to_csv(directory / "ambiguity_breakdown.csv", index=False)

    figures = [
        plots.model_comparison(_headline(table), directory / "model_comparison.png"),
        plots.per_class_recall(table, directory / "per_class_recall.png", class_names),
        plots.ambiguity_comparison(ambiguity, directory / "ambiguity_breakdown.png"),
    ]

    for name in models:
        model_directory = directory / name
        confusion = pd.read_csv(model_directory / "confusion.csv", index_col=0)
        figures.append(
            plots.confusion_heatmap(
                confusion, directory / f"confusion_{name}.png", f"{name}, all folds pooled"
            )
        )
        importance = model_directory / "feature_importance.csv"
        if importance.exists():
            figures.append(
                plots.feature_importance(
                    pd.read_csv(importance), directory / f"feature_importance_{name}.png"
                )
            )
        history = model_directory / "history.csv"
        if history.exists():
            figures.append(
                plots.training_history(
                    pd.read_csv(history), directory / f"training_history_{name}.png"
                )
            )
        occlusion = model_directory / "occlusion.csv"
        if occlusion.exists():
            figures.append(
                plots.occlusion_profile(
                    pd.read_csv(occlusion), directory / "occlusion.png", class_names
                )
            )

    print("\nMacro-F1 by model")
    print(_headline(table).round(4).to_string(index=False))
    print("\nMargin over the metadata control")
    print(margins.round(4).to_string(index=False))
    print("\nSplit by whether the native sample rate identifies the species")
    print(ambiguity.round(4).to_string(index=False))

    report = _write_markdown(cfg, table, margins, directory, figures, ambiguity)
    print(f"\n{len(figures)} figures and {report.name} written to {directory}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure()
    build(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
