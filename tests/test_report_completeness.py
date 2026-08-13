"""That the report says everything it claims to, and says it legibly.

Two failures sat in the committed reports for a long time and neither was visible to
a test. The table headed "Every comparison" held only the margins against each
family's declared control, so the strongest result in the project was absent from the
list the README points at. And every table was rounded to four decimal places before
being written, so four p values below 0.00005 were published as a literal `0`.

Both are read straight off the committed markdown here, because both were failures of
what the document says rather than of what the arithmetic computed.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.evaluate import families, figures, plots
from tests.conftest import REPORT_CONFIGS
from tests.helpers import needs

REPORTS = PROJECT_ROOT / "data" / "metadata" / "report"
CONFIG_FILE = REPORT_CONFIGS
CONFIGS = list(CONFIG_FILE)

# A markdown row, split on the pipes, with the outer empties dropped.
ROW = re.compile(r"^\|(.+)\|$")


def cells_of(line: str) -> list[str]:
    match = ROW.match(line.strip())
    return [cell.strip() for cell in match.group(1).split("|")] if match else []


def margins(name: str) -> pd.DataFrame:
    path = REPORTS / name / "family_margins.csv"
    if not path.exists():
        pytest.skip(f"{path.name} absent; run python -m src.evaluate.report first")
    return pd.read_csv(path)


def report(name: str) -> list[str]:
    path = REPORTS / name / "REPORT.md"
    if not path.exists():
        pytest.skip(f"{path.name} absent; run python -m src.evaluate.report first")
    return path.read_text(encoding="utf-8").splitlines()


ZERO = {"0", "0.0", "0.0000"}
SIGNIFICANCE = {"p_value", "q_value"}


@pytest.mark.parametrize("name", CONFIGS)
def test_no_p_value_is_published_as_zero(name):
    """A p value of exactly zero is a claim no test in this project can make.

    It reads the significance columns by position under their own header rather than
    scanning every cell of every wide table. Scanning found real zeros and called them
    rounding: under the wide place folds the metadata control has a recall of exactly
    zero on one class, and an ambiguity table counts a category that never occurs. Both
    are numbers this report should print as 0, and neither is a p value.
    """
    columns: dict[int, str] = {}
    offenders = []
    for line in report(name):
        cells = cells_of(line)
        if not cells:
            columns = {}
            continue
        if SIGNIFICANCE & set(cells):
            columns = {i: cell for i, cell in enumerate(cells) if cell in SIGNIFICANCE}
            continue
        offenders += [
            f"{columns[i]}={cell}" for i, cell in enumerate(cells) if i in columns and cell in ZERO
        ]
    assert not offenders, f"{name} publishes {sorted(offenders)} as zero"


@pytest.mark.parametrize("name", CONFIGS)
def test_every_margin_reaches_the_table_that_lists_every_margin(name):
    """Both floors, for every family that has two of them."""
    table = margins(name)
    assert "floor" in table.columns, "the overview cannot say what a margin was measured from"
    assert not table.duplicated(["family", "model", "floor"]).any()

    cfg = load_config(f"configs/{CONFIG_FILE[name]}")
    for family in families.discover(cfg):
        rows = table[table["family"] == family.key]
        assert family.control in set(rows["floor"]), f"{family.key} lost its declared control"

        strongest = families.strongest_floor(cfg, family)
        if strongest is not None and strongest != family.control:
            against = rows[rows["floor"] == strongest]
            assert set(against["model"]) == set(family.members) - {strongest}, (
                f"{family.key} is not fully measured against {strongest}"
            )


def test_the_headline_is_in_the_overview():
    """The comparison the README leads with, in the table the README points at.

    The margin is checked as a band rather than a value. Pinning the digit would make
    this fail on every refit, which is the opposite of useful: the claim is that an
    audio model loses to the paperwork by a quarter of a point on every fold, not that
    it loses by one particular number.
    """
    table = margins("base_10k")
    row = table[(table["model"] == "xgboost") & (table["floor"] == "logbook")]

    assert len(row) == 1, "xgboost against the logbook is missing from family_margins.csv"
    assert -0.35 < row["margin"].iloc[0] < -0.15, "the headline has moved out of its band"
    assert row["p_value"].iloc[0] < 1e-6, "rounding must not flatten this to zero"
    assert row["agreeing"].iloc[0] == row["folds"].iloc[0] == 50


def test_no_audio_model_beats_the_floor_in_any_configuration():
    """The finding, stated as a property of the artifacts rather than as prose.

    Four representations are tried against the strongest model that hears nothing. If
    one of them ever wins, this fails, and it should: that would be the result the
    project has spent its life looking for.
    """
    winners = []
    for name in CONFIGS:
        table = margins(name)
        floor = table[table["family"] == "species"]["floor"].unique()
        for basis in floor:
            rows = table[(table["family"] == "species") & (table["floor"] == basis)]
            winners += [
                f"{name}/{r.model} beats {basis} by {r.margin:+.3f} at p={r.p_value:.2g}"
                for r in rows.itertuples()
                if r.margin > 0 and r.p_value < 0.05 and r.model not in {"logbook", "metadata"}
            ]
    assert not winners, winners


def test_the_multiplicity_table_covers_every_configuration():
    path = REPORTS / "multiplicity.csv"
    needs(path, "run python -m src.evaluate.multiplicity first")
    table = pd.read_csv(path)

    assert set(table["config"]) == set(CONFIGS)
    assert len(table) == sum(len(margins(name)) for name in CONFIGS)
    assert (table["q_value"] >= table["p_value"]).all(), "a correction cannot be kinder"

    survivors = table[table["rejected"]]
    assert not survivors.empty, "the headline has to survive its own correction"
    assert (survivors["q_value"] < 0.05).all()


def test_the_headline_survives_the_correction():
    path = REPORTS / "multiplicity.csv"
    needs(path, "run python -m src.evaluate.multiplicity first")
    table = pd.read_csv(path).set_index(["config", "model", "floor"])
    row = table.loc[("base_10k", "xgboost", "logbook")]

    assert bool(row["rejected"])
    assert row["q_value"] < 0.05 / len(pd.read_csv(path)), "and clears the divided threshold too"


def test_the_headline_figure_draws_its_line_at_the_strongest_floor(monkeypatch):
    """The reference line was pinned to the metadata control by name for a long time.

    That drew it a tenth of a point below the score an audio model actually had to
    reach, in the one figure a reader looks at before any table.
    """
    cfg = load_config("configs/base.yaml")
    needs(REPORTS / cfg.name / "comparison.csv", "run python -m src.evaluate.report first")

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        figures.plots,
        "model_comparison",
        lambda table, path, **kwargs: seen.update(kwargs) or path,
    )
    monkeypatch.setattr(figures.plots, "per_class_recall", lambda *a, **k: a[1])
    monkeypatch.setattr(figures.plots, "confusion_heatmap", lambda *a, **k: a[1])
    monkeypatch.setattr(figures.plots, "ambiguity_comparison", lambda *a, **k: a[1])
    monkeypatch.setattr(
        figures,
        "_per_model_drawing",
        lambda *a, **k: {name: (lambda path, model: path) for name in figures.PER_MODEL},
    )

    family = next(f for f in families.discover(cfg) if f.key == families.SPECIES_FAMILY)
    comparison = pd.read_csv(REPORTS / cfg.name / "comparison.csv")
    ambiguity = pd.read_csv(REPORTS / cfg.name / "ambiguity_breakdown.csv")
    figures.draw_all(cfg, family, comparison, ambiguity)

    assert seen["floor"] == "logbook"
    assert "metadata" in seen["silent"]
    assert "xgboost" not in seen["silent"]


def test_each_giveaway_figure_is_titled_with_its_own_field(monkeypatch):
    """One title was hardcoded, so the collection code chart named the sample rate."""
    cfg = load_config("configs/base.yaml")
    needs(REPORTS / cfg.name / "comparison.csv", "run python -m src.evaluate.report first")

    fields: list[str] = []
    monkeypatch.setattr(figures.plots, "model_comparison", lambda *a, **k: a[1])
    monkeypatch.setattr(figures.plots, "per_class_recall", lambda *a, **k: a[1])
    monkeypatch.setattr(figures.plots, "confusion_heatmap", lambda *a, **k: a[1])
    monkeypatch.setattr(
        figures.plots,
        "ambiguity_comparison",
        lambda table, path, field: fields.append(field) or path,
    )
    monkeypatch.setattr(
        figures,
        "_per_model_drawing",
        lambda *a, **k: {name: (lambda path, model: path) for name in figures.PER_MODEL},
    )

    family = next(f for f in families.discover(cfg) if f.key == families.SPECIES_FAMILY)
    comparison = pd.read_csv(REPORTS / cfg.name / "comparison.csv")
    ambiguity = pd.read_csv(REPORTS / cfg.name / "ambiguity_breakdown.csv")
    figures.draw_all(cfg, family, comparison, ambiguity)

    assert fields == list(dict.fromkeys(ambiguity["giveaway"]))
    assert len(set(fields)) == len(fields), "two figures cannot carry the same title"


def test_the_plot_helpers_still_take_what_the_report_hands_them():
    """Guards the two signatures the fixes above depend on."""
    assert "field" in plots.ambiguity_comparison.__code__.co_varnames
    assert "floor" in plots.model_comparison.__code__.co_varnames
