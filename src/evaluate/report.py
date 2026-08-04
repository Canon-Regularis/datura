"""Assemble every result into one document, family by family.

A family is some models on trial and the control they were measured against. The
report leads with that gap rather than with the best score, because an audio model
that beats a control seeing no audio by two points has shown very little about
whale vocalisation, and that has to be visible without reading a table.

Every margin carries what the design can resolve beside it: the interval the paired
folds allow, the p value, and how many folds pointed the same way. Five folds over
a dozen recordings settle less than a mean and a standard deviation suggest.

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
from src.evaluate import families, plots, tables
from src.models import registry as models
from src.provenance import write as write_provenance
from src.results import config_directory, ensure, model_directory, report_path

logger = logging.getLogger(__name__)

# Committed tables get regenerated on another machine and diffed against what is in
# the repo, so a number has to survive the trip. Most of them do, because they are
# sums and divisions. The p value and its interval do not: scipy reaches them through
# the platform's libm, and the same comparison came out as 0.010063775865938413 on
# Windows and 0.010063775865938426 on Linux. That is a real byte difference and a
# meaningless numeric one, and it failed the build.
#
# Ten decimal places is five orders of magnitude coarser than that disagreement and
# still far more precision than any number here is read at. A p value written to
# seventeen significant figures was claiming a precision this design does not have,
# which is the argument the rest of the repo makes about everything else.
CSV_DECIMALS = 10

MARGIN_COLUMNS = (
    "Columns beside the margin say what the design resolves. `folds` counts every fold of "
    "every repeat, so a run of ten repeats over five folds shows 50. `low` and `high` bound "
    "the paired difference at 95%, and `p_value` is the corrected resampled test, which "
    "accounts for the training data those folds share. `agreeing` counts the folds that "
    "pointed the same way as the mean, and is worth reading where the p value settles "
    "nothing."
)


def _splits_behind(cfg: Config, name: str) -> int:
    """How many folds of how many repeats went into one model's pooled matrix.

    A confusion matrix pools every split, so a ten repeat run counts each clip ten
    times. The shares it is drawn from are unaffected, and the raw counts printed
    beside them are not, so the title says how many splits are behind the number.
    """
    return len(pd.read_csv(model_directory(cfg, name) / "fold_metrics_clip.csv"))


def _figures(
    cfg: Config,
    model_names: list[str],
    comparison: pd.DataFrame,
    ambiguity: pd.DataFrame,
) -> list[Path]:
    """Draw everything the species section links to.

    Per model figures are drawn only when the file behind them exists, so a partial
    run still produces a readable report.
    """
    directory = config_directory(cfg)
    class_names = list(cfg.dataset.species)

    figures = [
        plots.model_comparison(tables.headline(comparison), directory / "model_comparison.png"),
        plots.per_class_recall(comparison, directory / "per_class_recall.png", class_names),
    ]
    for field, rows in ambiguity.groupby("giveaway", sort=False):
        stem = field.replace(" ", "_")
        figures.append(
            plots.ambiguity_comparison(rows, directory / f"ambiguity_{stem}.png"),
        )

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
                f"{name}, {_splits_behind(cfg, name)} splits pooled",
            )
        )
        for filename, draw in optional.items():
            path = source / filename
            if path.exists():
                figures.append(draw(path, name))
    return figures


def _overview(margins: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every comparison in the configuration, most resolved first."""
    table = pd.concat(
        [frame.assign(family=key) for key, frame in margins.items()], ignore_index=True
    )
    columns = ["family", "model", "margin", "low", "high", "p_value", "agreeing", "folds"]
    return table[columns].sort_values("p_value").reset_index(drop=True)


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write one table at a precision that regenerates identically elsewhere.

    Every committed CSV goes through here, so none of them can reintroduce the
    platform dependent last digit that the reproduce job exists to catch.
    """
    frame.round(CSV_DECIMALS).to_csv(path, index=False)
    return path


def _no_audio_models() -> set[str]:
    """Every registered model that is never given the recording."""
    return {spec.name for spec in models.specs() if not spec.hears_audio}


def _strongest_floor_section(cfg: Config, family: families.Family) -> list[str]:
    """Every model against the best a model with no audio manages.

    The metadata control was built before anyone knew what else the paperwork
    carried, so it is a floor rather than the floor. Reporting the higher one beside
    it says how much of the gap was equipment and how much was everything else
    written down about the recording.
    """
    floors = [name for name in family.names if name in _no_audio_models()]
    if len(floors) < 2:
        return []

    strongest = _best_floor(cfg, family, floors)
    if strongest == family.control:
        return []

    against = tables.family_margins(cfg, family, control=strongest)
    return [
        f"### Margin over {strongest}, the strongest model that hears no audio",
        "",
        f"`{strongest}` also sees the site, the coordinates, the noise conditions and the",
        "collection code the field note opens with. None of that is the animal, so this is the",
        "number an audio result has to clear before it is evidence about whales.",
        "",
        against.round(4).to_markdown(index=False),
        "",
    ]


def _best_floor(cfg: Config, family: families.Family, floors: list[str]) -> str:
    """Whichever no-audio model scores highest, which is the real floor."""
    scores = {
        name: pd.read_csv(model_directory(cfg, name) / "summary.csv")
        .query("metric == 'macro_f1'")["mean"]
        .iloc[0]
        for name in floors
    }
    return max(scores, key=scores.get)


def _species_section(
    cfg: Config,
    family: families.Family,
    comparison: pd.DataFrame,
    margins: pd.DataFrame,
    intervals: pd.DataFrame,
    ambiguity: pd.DataFrame,
    figures: list[Path],
) -> list[str]:
    class_names = list(cfg.dataset.species)
    return [
        f"## {family.title}",
        "",
        "### Margin over the metadata control",
        "",
        "The control sees native sample rate, year, clip duration and file size; it sees no",
        "audio. Its score is the floor an audio model has to clear.",
        "",
        margins.round(4).to_markdown(index=False),
        "",
        *_strongest_floor_section(cfg, family),
        "### Every model, with the range the recordings support",
        "",
        "The interval comes from resampling whole tapes with replacement. Cuts from one tape",
        "are near duplicates, so resampling clips would count the same recording many times",
        "and produce an interval several times too narrow.",
        "",
        intervals.round(4).to_markdown(index=False),
        "",
        "### Spread across folds",
        "",
        tables.headline(comparison).round(4).to_markdown(index=False),
        "",
        "### Per species recall",
        "",
        tables.per_species_recall(comparison, class_names).round(4).to_markdown(index=False),
        "",
        "### With and without the equipment giveaway",
        "",
        "Test clips split by whether their native sample rate is used by one species or by",
        "several. On the shared rate subset the recording cannot identify the species by",
        "itself, so that column is where audio has to earn its result.",
        "",
        ambiguity.round(4).to_markdown(index=False),
        "",
        "### Figures",
        "",
        *[f"- `{path.name}`" for path in figures],
        "",
        "Every figure has a CSV of the same name beside it, or in the model directory.",
        "",
    ]


def _call_type_section(
    family: families.Family, margins: pd.DataFrame, intervals: pd.DataFrame
) -> list[str]:
    return [
        f"## {family.title}",
        "",
        f"Against `{family.control}`, which sees the site, the coordinates and the noise",
        "conditions, and no audio.",
        "",
        margins.round(4).to_markdown(index=False),
        "",
        intervals.round(4).to_markdown(index=False),
        "",
    ]


def _header(cfg: Config, family_count: int) -> list[str]:
    class_names = list(cfg.dataset.species)
    return [
        f"# Results: {cfg.name}",
        "",
        f"Species: {', '.join(class_names)}  ",
        f"Common band: 0 to {cfg.audio.nyquist:.0f} Hz at {cfg.audio.target_sample_rate} Hz  ",
        f"Folds: {cfg.split.n_folds} per split, grouped by tape  ",
        f"Windows: {cfg.audio.window_seconds} s, hop {cfg.audio.hop_seconds} s, "
        f"at most {cfg.audio.max_windows_per_clip} per clip  ",
        f"Families: {family_count}, each a set of models and the control they were measured "
        "against",
        "",
    ]


def build(cfg: Config) -> Path:
    """Write every family's tables, the species figures, and the summary."""
    directory = ensure(cfg)
    discovered = families.discover(cfg)
    if not discovered:
        raise tables.MissingResults(
            f"no results under {directory}; run python -m src.pipeline --config {cfg.source.name}"
        )

    margins = {family.key: tables.family_margins(cfg, family) for family in discovered}
    intervals = {family.key: tables.family_intervals(cfg, family) for family in discovered}

    overview = _overview(margins)
    write_table(overview, directory / "family_margins.csv")

    sections = [
        *_header(cfg, len(discovered)),
        "## Every comparison",
        "",
        MARGIN_COLUMNS,
        "",
        overview.round(4).to_markdown(index=False),
        "",
    ]

    figures: list[Path] = []
    for family in discovered:
        if family.key != families.SPECIES_FAMILY:
            sections += _call_type_section(family, margins[family.key], intervals[family.key])
            continue

        comparison = tables.comparison(cfg, list(family.names))
        write_table(comparison, directory / "comparison.csv")
        write_table(margins[family.key], directory / "margin_over_control.csv")

        ambiguity = tables.ambiguity(cfg, list(family.names))
        write_table(ambiguity, directory / "ambiguity_breakdown.csv")

        figures = _figures(cfg, list(family.names), comparison, ambiguity)
        sections += _species_section(
            cfg,
            family,
            comparison,
            margins[family.key],
            intervals[family.key],
            ambiguity,
            figures,
        )

    resolved = overview[overview["p_value"] < 0.05]
    logger.info("\nEvery comparison, most resolved first\n%s", overview.round(4).to_string())
    logger.info("%d of %d comparisons are resolved at p < 0.05", len(resolved), len(overview))

    write_provenance(cfg, directory, extra={"families": [family.key for family in discovered]})
    path = report_path(cfg)
    path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("\n%d figures and %s written to %s", len(figures), path.name, directory)
    return path


def main(argv: list[str] | None = None) -> int:
    args = cli.parser_for(__doc__).parse_args(argv)
    build(cli.prepare(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
