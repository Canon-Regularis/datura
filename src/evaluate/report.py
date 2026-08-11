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
from src.evaluate import coverage, ensemble, families, sections, tables
from src.evaluate import figures as figure_set
from src.evaluate.artifacts import write_table
from src.provenance import write as write_provenance
from src.results import ensure, report_path

logger = logging.getLogger(__name__)


def _shared_tapes(cfg: Config) -> pd.DataFrame | None:
    """Tapes carrying more than one species, when the audit has been written."""
    path = cfg.paths.metadata / f"audit_cross_species_tapes_{cfg.corpus}.csv"
    return pd.read_csv(path) if path.exists() else None


def build(cfg: Config) -> Path:
    """Write every family's tables, the species figures, and the summary."""
    directory = ensure(cfg)

    # Before discovery, because the averaged model has to be on disk to be found. It
    # refits nothing: the members were scored on identical folds, so their per clip
    # probabilities join and average.
    ensemble.materialise(cfg)

    discovered = families.discover(cfg)
    if not discovered:
        raise tables.MissingResults(
            f"no results under {directory}; run python -m src.pipeline --config {cfg.source.name}"
        )

    # Two floors per family where they differ. A model is measured against the control
    # it was declared with, and again against the best any model that hears nothing
    # manages. Only the first used to reach this table, which is how the strongest
    # comparison in this project stayed out of the list that claims to hold every one.
    margins: dict[str, pd.DataFrame] = {}
    against_floor: dict[str, pd.DataFrame] = {}
    for family in discovered:
        declared = tables.family_margins(cfg, family)
        margins[family.key] = declared.assign(family=family.key, floor=family.control)

        strongest = families.strongest_floor(cfg, family)
        if strongest is not None and strongest != family.control:
            against = tables.family_margins(cfg, family, control=strongest)
            against_floor[family.key] = against.assign(family=family.key, floor=strongest)

    intervals = {family.key: tables.family_intervals(cfg, family) for family in discovered}

    overview = sections.overview([*margins.values(), *against_floor.values()])
    write_table(overview, directory / "family_margins.csv")

    document = [
        *sections.header(cfg, len(discovered)),
        "## Every comparison",
        "",
        sections.MARGIN_COLUMNS,
        "",
        sections.markdown(overview),
        "",
    ]

    figures: list[Path] = []
    for family in discovered:
        if family.key != families.SPECIES_FAMILY:
            document += sections.call_type_section(
                family, margins[family.key], intervals[family.key]
            )
            continue

        comparison = tables.comparison(cfg, list(family.names))
        write_table(comparison, directory / "comparison.csv")
        write_table(
            margins[family.key].drop(columns=["family", "floor"]),
            directory / "margin_over_control.csv",
        )

        ambiguity = tables.ambiguity(cfg, list(family.names))
        write_table(ambiguity, directory / "ambiguity_breakdown.csv")

        # What each audio model is worth when it is allowed to decline. The controls
        # are left out: an operating curve for the logbook would describe how
        # confidently it reads paperwork, which is not a question anyone asks of a
        # tool that takes a wav file.
        heard = [name for name in family.names if name not in family.floors]
        operating = coverage.table(cfg, heard)
        write_table(operating, directory / "coverage.csv")

        figures = figure_set.draw_all(cfg, family, comparison, ambiguity, operating)
        document += sections.species_section(
            cfg,
            family,
            comparison,
            margins[family.key],
            intervals[family.key],
            ambiguity,
            figures,
            _shared_tapes(cfg),
            against_floor.get(family.key),
            operating,
        )

    resolved = overview[overview["p_value"] < 0.05]
    logger.info("\nEvery comparison, most resolved first\n%s", overview.to_string())
    logger.info(
        "%d of %d comparisons are resolved at an uncorrected p < 0.05; "
        "run python -m src.evaluate.multiplicity for the adjusted verdict",
        len(resolved),
        len(overview),
    )

    write_provenance(cfg, directory, extra={"families": [family.key for family in discovered]})
    path = report_path(cfg)
    path.write_text("\n".join(document), encoding="utf-8")
    logger.info("\n%d figures and %s written to %s", len(figures), path.name, directory)
    return path


def main(argv: list[str] | None = None) -> int:
    args = cli.parser_for(__doc__).parse_args(argv)
    build(cli.prepare(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
