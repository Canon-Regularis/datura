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

from src import cli
from src.config import Config
from src.evaluate import families, sections, tables
from src.evaluate import figures as figure_set
from src.evaluate.artifacts import write_table
from src.provenance import write as write_provenance
from src.results import ensure, report_path

logger = logging.getLogger(__name__)


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

    overview = sections.overview(margins)
    write_table(overview, directory / "family_margins.csv")

    document = [
        *sections.header(cfg, len(discovered)),
        "## Every comparison",
        "",
        sections.MARGIN_COLUMNS,
        "",
        overview.round(4).to_markdown(index=False),
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
        write_table(margins[family.key], directory / "margin_over_control.csv")

        ambiguity = tables.ambiguity(cfg, list(family.names))
        write_table(ambiguity, directory / "ambiguity_breakdown.csv")

        figures = figure_set.draw_all(cfg, list(family.names), comparison, ambiguity)
        document += sections.species_section(
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
    path.write_text("\n".join(document), encoding="utf-8")
    logger.info("\n%d figures and %s written to %s", len(figures), path.name, directory)
    return path


def main(argv: list[str] | None = None) -> int:
    args = cli.parser_for(__doc__).parse_args(argv)
    build(cli.prepare(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
