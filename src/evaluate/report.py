"""Assemble every model's results into one comparison, with figures and a summary.

The summary leads with the metadata control rather than with the best score. An
audio model that beats the control by two points has shown very little about whale
vocalisation, and that has to be visible without reading a table.

Tables come from ``src.evaluate.tables``; this module only arranges and writes
them.

Usage:
    python -m src.evaluate.report [--config configs/base.yaml]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src import cli
from src.config import Config
from src.evaluate import plots, tables
from src.provenance import write as write_provenance
from src.results import config_directory, ensure, model_directory, report_path

logger = logging.getLogger(__name__)


def _figures(
    cfg: Config,
    model_names: list[str],
    comparison: pd.DataFrame,
    ambiguity: pd.DataFrame,
) -> list[Path]:
    """Draw everything the report links to.

    Per model figures are drawn only when the file behind them exists, so a partial
    run still produces a readable report.
    """
    directory = config_directory(cfg)
    class_names = list(cfg.dataset.species)

    figures = [
        plots.model_comparison(tables.headline(comparison), directory / "model_comparison.png"),
        plots.per_class_recall(comparison, directory / "per_class_recall.png", class_names),
        plots.ambiguity_comparison(ambiguity, directory / "ambiguity_breakdown.png"),
    ]

    optional = {
        "feature_importance.csv": lambda path, name: plots.feature_importance(
            pd.read_csv(path), directory / f"feature_importance_{name}.png"
        ),
        "history.csv": lambda path, name: plots.training_history(
            pd.read_csv(path), directory / f"training_history_{name}.png"
        ),
        "occlusion.csv": lambda path, name: plots.occlusion_profile(
            pd.read_csv(path), directory / "occlusion.png", class_names
        ),
    }

    for name in model_names:
        source = model_directory(cfg, name)
        figures.append(
            plots.confusion_heatmap(
                pd.read_csv(source / "confusion.csv", index_col=0),
                directory / f"confusion_{name}.png",
                f"{name}, all folds pooled",
            )
        )
        for filename, draw in optional.items():
            path = source / filename
            if path.exists():
                figures.append(draw(path, name))
    return figures


def _markdown(
    cfg: Config,
    comparison: pd.DataFrame,
    margins: pd.DataFrame,
    ambiguity: pd.DataFrame,
    figures: list[Path],
) -> str:
    class_names = list(cfg.dataset.species)
    sections = [
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
        "The control sees native sample rate, year, clip duration and file size; it sees no",
        "audio. Its score is the floor an audio model has to clear.",
        "",
        margins.round(4).to_markdown(index=False),
        "",
        "## All models",
        "",
        tables.headline(comparison).round(4).to_markdown(index=False),
        "",
        "## Per species recall",
        "",
        tables.per_species_recall(comparison, class_names).round(4).to_markdown(index=False),
        "",
        "## With and without the equipment giveaway",
        "",
        "Test clips split by whether their native sample rate is used by one species or by",
        "several. On the shared rate subset the recording cannot identify the species by",
        "itself, so that column is where audio has to earn its result.",
        "",
        ambiguity.round(4).to_markdown(index=False),
        "",
        "## Figures",
        "",
        *[f"- `{path.name}`" for path in figures],
        "",
        "Every figure has a CSV of the same name beside it, or in the model directory.",
        "",
    ]
    return "\n".join(sections)


def build(cfg: Config) -> Path:
    """Write the comparison, the figures and the summary for one configuration."""
    directory = ensure(cfg)
    model_names = tables.available_models(cfg)
    if not model_names:
        raise tables.MissingResults(
            f"no results under {directory}; run python -m src.pipeline --config {cfg.source.name}"
        )

    comparison = tables.comparison(cfg, model_names)
    comparison.to_csv(directory / "comparison.csv", index=False)

    margins = tables.margin_over_control(comparison)
    margins.to_csv(directory / "margin_over_control.csv", index=False)

    ambiguity = tables.ambiguity(cfg, model_names)
    ambiguity.to_csv(directory / "ambiguity_breakdown.csv", index=False)

    figures = _figures(cfg, model_names, comparison, ambiguity)

    logger.info(
        "\nMacro-F1 by model\n%s", tables.headline(comparison).round(4).to_string(index=False)
    )
    logger.info("\nMargin over the metadata control\n%s", margins.round(4).to_string(index=False))
    logger.info(
        "\nSplit by whether the native sample rate identifies the species\n%s",
        ambiguity.round(4).to_string(index=False),
    )

    write_provenance(cfg, directory, extra={"models": model_names})
    path = report_path(cfg)
    path.write_text(_markdown(cfg, comparison, margins, ambiguity, figures), encoding="utf-8")
    logger.info("\n%d figures and %s written to %s", len(figures), path.name, directory)
    return path


def main(argv: list[str] | None = None) -> int:
    args = cli.parser_for(__doc__).parse_args(argv)
    build(cli.prepare(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
