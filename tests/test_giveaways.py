"""What a recorded field does to the species, and what a missing one does not.

Splitting held out clips on whether a field gives the species away is the argument
this project rests its confound claim on, and it was wrong in two ways at once.

A blank collection code counted as a value, so the clips carrying no code landed in
the bucket for codes that several species share. In the three species set that bucket
was the only one of its kind, because no code there is shared at all, so a table
labelled as though a giveaway had been measured and found ambiguous was reporting the
opposite.

The scores in it were capped as well. A slice holding two of three species is scored
over three, which divides by a class that cannot appear and puts two thirds out of
reach before the model is consulted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.data.audit import context as audit
from src.scoring import from_counts

METADATA = PROJECT_ROOT / "data" / "metadata"


def corpus(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(f"configs/{name}")
    manifest = METADATA / f"manifest_{cfg.name}.parquet"
    notes = METADATA / "watkins_annotations.parquet"
    if not manifest.exists() or not notes.exists():
        pytest.skip(f"{manifest.name} absent; run python -m src.data.manifest first")
    kept = pd.read_parquet(manifest)
    return kept[kept["keep"]], pd.read_parquet(notes)


def frame(codes: list[str], species: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clip_id": [f"c{i}" for i in range(len(codes))],
            "collection_code": codes,
            "species": species,
        }
    )


def test_a_blank_is_not_a_value_two_species_share():
    table = frame(["AA", "AA", "", ""], ["one", "one", "one", "two"])
    assert audit.shared_values(table, "collection_code") == set()


def test_a_code_two_species_use_is_shared():
    table = frame(["AA", "AA", "BB"], ["one", "two", "two"])
    assert audit.shared_values(table, "collection_code") == {"AA"}


def test_a_missing_value_gets_its_own_label():
    table = frame(["AA", "BB", ""], ["one", "one", "two"])
    table["native_sample_rate"] = [10000, 10000, 20000]
    labels = audit.giveaway_labels(table)["collection code"]

    assert labels.loc["c0"] == audit.UNIQUE
    assert labels.loc["c2"] == audit.ABSENT
    assert audit.SHARED not in set(labels)


def test_a_numeric_field_is_never_read_as_blank():
    """A sample rate of zero would be a value. Only text can be absent."""
    table = frame(["AA", "BB"], ["one", "two"])
    table["native_sample_rate"] = [0, 0]
    labels = audit.giveaway_labels(table)["native sample rate"]
    assert set(labels) == {audit.SHARED}


def test_no_code_in_the_narrow_set_is_shared_by_two_species():
    kept, notes = corpus("base.yaml")
    joined = audit.with_columns(kept, notes, "collection_code")
    assert audit.shared_values(joined, "collection_code") == set()


def test_the_clips_with_no_code_are_a_handful_of_recordings():
    """359 clips, eight tapes, two of the three species. The bucket the README quotes."""
    kept, notes = corpus("base.yaml")
    labels = audit.giveaway_labels(kept, notes)["collection code"]
    absent = kept[kept["clip_id"].isin(labels[labels == audit.ABSENT].index)]

    assert len(absent) == 359
    assert absent["tape_id"].nunique() == 8
    assert set(absent["species"]) == {"KillerWhale", "SpermWhale"}


def test_the_wide_set_does_carry_shared_codes():
    """The three species set cannot ask the question. Eleven species can."""
    kept, notes = corpus("wide.yaml")
    joined = audit.with_columns(kept, notes, "collection_code")
    assert audit.shared_values(joined, "collection_code") == {"BA2A", "BE3B", "BE3C"}


def test_an_empty_site_is_not_a_place():
    kept, notes = corpus("base.yaml")
    row = audit.site_giveaway(kept, notes).iloc[0]

    assert row["sites"] == 47, "the blank used to be counted as a forty eighth site"
    assert row["clips_carrying_a_site"] == 4088
    assert row["clips"] == 4160


def test_a_missing_class_caps_the_macro_average():
    """Two of three species present, so a three class average cannot pass two thirds."""
    labels = np.array([0, 0, 1, 1])
    perfect = np.array([0, 0, 1, 1])

    over_all_three = from_counts(labels, perfect, 3)["macro_f1"]
    over_the_two_present = from_counts(labels, perfect, 3, np.array([0, 1]))["macro_f1"]

    assert over_all_three == pytest.approx(2 / 3)
    assert over_the_two_present == pytest.approx(1.0)


def test_restricting_the_average_leaves_the_prediction_alone():
    """A class the slice does not hold can still be predicted, and it still costs."""
    labels = np.array([0, 0, 1, 1])
    strays = np.array([0, 2, 1, 1])

    scored = from_counts(labels, strays, 3, np.array([0, 1]))
    assert scored["macro_f1"] < 1.0
    assert scored["accuracy"] == pytest.approx(0.75)
