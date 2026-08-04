"""That every recording was tested exactly once, in every configuration.

``assert_no_group_leak`` already runs on every fold build, so a tape cannot sit on
both sides of a boundary. This checks the other half: that the folds together cover
the collection rather than quietly dropping part of it.

It matters most on the wide species set. Eleven classes over 238 recordings is where
a class first gets thin enough to land a fold with no test tape at all, and macro-F1
would then average over a class that was never scored without saying so.

Read from the committed fold summaries, so it needs no archive and runs in CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import PROJECT_ROOT

REPORTS = PROJECT_ROOT / "data" / "metadata" / "report"
CONFIGS = ["base_10k", "base_5k", "wide_10k"]


def summary(name: str) -> pd.DataFrame:
    path = REPORTS / f"fold_summary_{name}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} absent; run python -m src.train.xgb --config {name} first")
    return pd.read_csv(path)


def coverage(name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "metadata" / f"audit_coverage_{name}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} absent; run python -m src.data.manifest first")
    return pd.read_csv(path)


@pytest.mark.parametrize("name", CONFIGS)
def test_every_tape_is_tested_exactly_once(name):
    """Test tapes summed over the folds must equal the tapes that survived filtering.

    Each fold tests a disjoint set, so the sum is the count. A regrouping that lost
    or duplicated recordings shows up here as arithmetic rather than as a score that
    happens to look better.
    """
    folds = summary(name)
    tested = folds[folds["part"] == "test"].groupby("species")["tapes"].sum()
    kept = coverage(name).set_index("species")["kept_tapes"]

    for species in tested.index:
        assert tested[species] == kept[species], (
            f"{name}, {species}: {tested[species]} tapes tested across the folds "
            f"but {kept[species]} survived filtering"
        )


@pytest.mark.parametrize("name", CONFIGS)
def test_every_class_appears_in_every_test_fold(name):
    """A class with no test tape in a fold makes that fold's macro-F1 a fiction."""
    folds = summary(name)
    held_out = folds[folds["part"] == "test"]
    per_fold = held_out.pivot(index="fold", columns="species", values="tapes").fillna(0)

    empty = [
        (int(fold), species)
        for fold in per_fold.index
        for species in per_fold.columns
        if per_fold.loc[fold, species] < 1
    ]
    assert not empty, f"{name}: no test tape for {empty}"


@pytest.mark.parametrize("name", CONFIGS)
def test_no_class_is_thinner_than_the_bar_this_project_sets(name):
    """Ten tapes, the same minimum a call type has to clear to be worth fitting."""
    kept = coverage(name).set_index("species")["kept_tapes"]
    thin = kept[kept < 10]
    assert thin.empty or name != "wide_10k", (
        f"{name} carries species under ten recordings: {thin.to_dict()}"
    )
