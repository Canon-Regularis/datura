"""Reading call types and sites out of free prose.

The Watkins notes were written for people, not for a parser, so the risk here is
silent mislabelling: a note that quietly contributes the wrong label is worse than
one that contributes none. These tests pin the cases that decide that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.annotations import (
    AnnotationError,
    annotate,
    attach_context,
    call_columns,
    condition_columns,
    context_columns,
)
from src.data.notes import CALL_PREFIX, CONDITION_PREFIX, Vocabulary, tag_note

VOCABULARY = Vocabulary(
    call_types={
        "pulsed_call": ["pulsed call"],
        "click": ["click"],
        "whistle": ["whistle"],
        "call": ["call"],
        "squeal": ["squeal"],
    },
    conditions={
        "ship_noise": ["ship noise", "propeller"],
        "reverberation": ["reverberation"],
    },
)


def test_reads_several_call_types_from_one_note():
    calls, _ = tag_note("BE7A  Squeal; chirp.  Reverberation present.", VOCABULARY)
    assert calls == {"squeal"}


def test_a_note_can_carry_more_than_one_call_type():
    calls, _ = tag_note("BA2A  Clicks; whistle (Delphinidae).", VOCABULARY)
    assert calls == {"click", "whistle"}


def test_a_longer_phrase_wins_over_the_shorter_one_inside_it():
    """ "Pulsed calls" is a pulsed call, and must not also register as a call."""
    calls, _ = tag_note("AC2A  Calls; pulsed calls.  SOFAR recordings.", VOCABULARY)
    assert "pulsed_call" in calls
    assert "call" in calls, "the plain 'Calls' earlier in the note is a genuine second match"

    only_pulsed, _ = tag_note("AC2A  Pulsed call.", VOCABULARY)
    assert only_pulsed == {"pulsed_call"}


def test_conditions_are_kept_apart_from_call_types():
    calls, conditions = tag_note("BA2A  Clicks; ship noise.", VOCABULARY)
    assert calls == {"click"}
    assert conditions == {"ship_noise"}


def test_matching_ignores_case_and_plurals():
    assert tag_note("CLICKS", VOCABULARY)[0] == {"click"}
    assert tag_note("Click", VOCABULARY)[0] == {"click"}


@pytest.mark.parametrize("note", [None, "", "BE7A  OCA  Tape # 407", "No Visual"])
def test_notes_with_no_call_described_yield_nothing(note):
    """Logistics notes are common. Forcing a label out of them would invent data."""
    calls, _ = tag_note(note, VOCABULARY)
    assert calls == set()


def test_annotate_builds_one_row_per_clip_with_flag_columns():
    metadata = pd.DataFrame(
        [
            {
                "record_number": "52008001",
                "display_name": "SpermWhale",
                "note": "BA2A  Clicks; ship noise.",
                "location": {"name": ["Bermuda"], "coordinates": [{"lat": 32.0, "lon": -64.0}]},
                "observation_date": None,
            }
        ]
    )
    frame = annotate(metadata, VOCABULARY)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["clip_id"] == "52008001"
    assert row["site"] == "Bermuda"
    assert row["latitude"] == 32.0
    assert row["longitude"] == -64.0
    assert row[f"{CALL_PREFIX}click"]
    assert not row[f"{CALL_PREFIX}whistle"]
    assert row[f"{CONDITION_PREFIX}ship_noise"]
    assert row["n_call_types"] == 1


def test_site_survives_arrow_handing_back_numpy_arrays():
    """``to_pandas`` turns the nested lists into arrays, which once emptied every site."""
    metadata = pd.DataFrame(
        [
            {
                "record_number": "51041C01",
                "display_name": "BottlenoseDolphin",
                "note": "Whistles.",
                "location": {
                    "name": np.array(["Biscayne Bay, Florida"], dtype=object),
                    "coordinates": np.array([{"lat": 25.0, "lon": -80.0}], dtype=object),
                },
                "observation_date": None,
            }
        ]
    )
    row = annotate(metadata, VOCABULARY).iloc[0]
    assert row["site"] == "Biscayne Bay, Florida"
    assert row["latitude"] == 25.0


def test_a_missing_location_is_empty_rather_than_an_error():
    metadata = pd.DataFrame(
        [
            {
                "record_number": "49001001",
                "display_name": "Beluga_WhiteWhale",
                "note": "Squeals.",
                "location": None,
                "observation_date": None,
            }
        ]
    )
    row = annotate(metadata, VOCABULARY).iloc[0]
    assert row["site"] == ""
    assert row["latitude"] is None or pd.isna(row["latitude"])


def test_column_helpers_select_the_right_prefixes():
    frame = annotate(
        pd.DataFrame(
            [
                {
                    "record_number": "1",
                    "display_name": "X",
                    "note": "Clicks.",
                    "location": None,
                    "observation_date": None,
                }
            ]
        ),
        VOCABULARY,
    )
    assert set(call_columns(frame)) == {f"{CALL_PREFIX}{n}" for n in VOCABULARY.call_labels}
    assert set(condition_columns(frame)) == {
        f"{CONDITION_PREFIX}{n}" for n in VOCABULARY.condition_labels
    }
    assert not set(call_columns(frame)) & set(condition_columns(frame))


def context_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clip_id": ["5401800A", "54018001"],
            "site": ["Bermuda", ""],
            "latitude": [32.3, None],
            "longitude": [-64.8, None],
            "collection_code": ["BA2A", ""],
            "cond_ship_noise": [True, False],
            "call_click": [True, True],
        }
    )


def test_context_columns_are_the_circumstances_and_not_the_call():
    columns = condition_columns(context_frame())
    context = context_columns(context_frame())

    assert context == ["site", "latitude", "longitude", "collection_code", *columns]
    assert not any(name.startswith(CALL_PREFIX) for name in context), (
        "a call type describes the animal, so no control may see one"
    )


def test_attaching_context_joins_every_circumstance_and_nothing_else():
    """One list, one merge. Three copies of it is how a field goes unmeasured."""
    parsed = context_frame()
    index = pd.DataFrame(
        {"clip_id": ["5401800A", "5401800A", "54018001"], "window_index": [0, 1, 0]}
    )

    joined = attach_context(index, parsed)

    assert list(joined.columns) == ["clip_id", "window_index", *context_columns(parsed)]
    assert len(joined) == 3, "one row per window, not per clip"
    assert joined["collection_code"].tolist() == ["BA2A", "BA2A", ""]
    assert CALL_PREFIX + "click" not in joined.columns


def test_attaching_context_refuses_notes_that_were_parsed_without_it():
    stale = context_frame().drop(columns=["collection_code"])
    index = pd.DataFrame({"clip_id": ["5401800A"]})

    with pytest.raises(AnnotationError, match="collection_code"):
        attach_context(index, stale)
