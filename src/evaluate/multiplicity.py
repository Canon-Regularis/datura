"""What survives once every comparison in the project is counted at once.

Each configuration reports its margins with an uncorrected p value, which is the
right number for one comparison read on its own. This project does not report one.
It reports two dozen, ranks them, and calls the smallest a finding, and at 0.05
apiece a set that size expects more than one to clear the bar carrying nothing.

The family is the union of every configuration rather than each one separately.
Correcting inside a configuration would give the same comparison a different adjusted
value depending on which file it was read from, and would invite quoting the smaller.

Usage:
    python -m src.evaluate.multiplicity [--config configs/base.yaml ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import uncertainty
from src.config import Config, load_config
from src.errors import DaturaError
from src.evaluate.artifacts import write_table
from src.evaluate.sections import markdown
from src.results import family_margins_path

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = (
    "configs/base.yaml",
    "configs/base_5k.yaml",
    "configs/wide.yaml",
    "configs/context.yaml",
    "configs/context_shuffled.yaml",
)
THRESHOLD = 0.05

COLUMNS = (
    "config",
    "family",
    "model",
    "floor",
    "margin",
    "p_value",
    "q_value",
    "rejected",
    "agreeing",
    "folds",
)


class MissingMargins(DaturaError):
    """Raised when a configuration has no margins to correct."""


def gather(configs: list[Config]) -> pd.DataFrame:
    """Every reported comparison, from every configuration, in one frame.

    Every configuration is required. Reading whatever happens to be on disk would
    make the correction depend on which reports had been rebuilt most recently, so a
    missing one is an error rather than a smaller family.
    """
    frames = []
    for cfg in configs:
        path = family_margins_path(cfg)
        if not path.exists():
            raise MissingMargins(
                f"{path} is missing; run python -m src.evaluate.report --config {cfg.source.name}"
            )
        frames.append(pd.read_csv(path).assign(config=cfg.name))
    return pd.concat(frames, ignore_index=True)


def adjust(margins: pd.DataFrame) -> pd.DataFrame:
    """The same comparisons with a false discovery rate added, most significant first."""
    table = margins.copy()
    table["q_value"] = uncertainty.benjamini_hochberg(table["p_value"].to_numpy())
    table["rejected"] = table["q_value"] < THRESHOLD
    columns = [name for name in COLUMNS if name in table.columns]
    return table[columns].sort_values("q_value", kind="stable").reset_index(drop=True)


def _document(table: pd.DataFrame, configs: list[Config]) -> list[str]:
    survivors = table[table["rejected"]]
    named = ", ".join(cfg.name for cfg in configs)
    bonferroni = THRESHOLD / len(table)

    lines = [
        "# Every comparison, corrected for how many there are",
        "",
        f"{len(table)} comparisons across {named}. Each configuration reports these with an "
        "uncorrected p value, which is the right number to read for one comparison on its own "
        "and the wrong one to read across a table of them.",
        "",
        "`q_value` is the Benjamini-Hochberg adjusted figure, controlling the share of "
        "resolved comparisons that are noise. `rejected` marks the ones that survive it at "
        f"{THRESHOLD}. For reference, dividing the threshold across every comparison instead "
        f"puts the bar at {bonferroni:.2e}, which is the stricter question of whether any one "
        "of them is a false positive.",
        "",
        markdown(table),
        "",
        "## What survives",
        "",
    ]

    if survivors.empty:
        lines += ["Nothing clears the adjusted threshold.", ""]
        return lines

    lines += [
        f"{len(survivors)} of {len(table)} comparisons survive the correction, and the smallest "
        f"of them clears the divided threshold of {bonferroni:.2e} as well.",
        "",
    ]
    for _, row in survivors.iterrows():
        direction = "above" if row["margin"] >= 0 else "below"
        lines.append(
            f"- `{row['model']}` is {abs(row['margin']):.3f} {direction} `{row['floor']}` in "
            f"{row['config']}, at q = {row['q_value']:.2e} over {int(row['folds'])} folds."
        )
    lines.append("")
    return lines


def build(configs: list[Config]) -> Path:
    """Write the corrected table and the document beside the per configuration reports."""
    table = adjust(gather(configs))
    root = configs[0].paths.reports
    root.mkdir(parents=True, exist_ok=True)

    write_table(table, root / "multiplicity.csv")
    path = root / "MULTIPLICITY.md"
    path.write_text("\n".join(_document(table, configs)), encoding="utf-8")

    logger.info(
        "%d of %d comparisons survive Benjamini-Hochberg at %s",
        int(table["rejected"].sum()),
        len(table),
        THRESHOLD,
    )
    logger.info("\n%s", table.to_string(index=False))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="a configuration to include; repeat it, or omit for all three",
    )
    args = parser.parse_args(argv)
    build([load_config(name) for name in (args.configs or DEFAULT_CONFIGS)])
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
