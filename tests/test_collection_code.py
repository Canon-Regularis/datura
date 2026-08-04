"""Reading the collection code off the front of a note.

This one string turned out to identify the species better than the audio does, so
what it matches has to be pinned exactly. A regex that quietly caught a word from
the prose would manufacture the whole finding.
"""

from __future__ import annotations

import pytest

from src.data.notes import collection_code


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("BE7A   Squeal.  Reverberation present.  Good cut.", "BE7A"),
        ("BA2A  X  Clicks; ship noise.", "BA2A"),
        ("AC2A  Growl.  **Tape speed 30 ips.", "AC2A"),
        ("BD19D  Whistles; clicks.", "BD19D"),
        ("  BE3B  X  N15 13.92  W061 32.39.", "BE3B"),
    ],
)
def test_it_reads_the_code_a_note_opens_with(note, expected):
    assert collection_code(note) == expected


@pytest.mark.parametrize(
    "note",
    [
        "Clicks; ship noise.",
        "Two calls; clicks.",
        "The killer whales were quite spread out.",
        "",
        None,
    ],
)
def test_a_note_without_a_leading_code_reports_nothing(note):
    assert collection_code(note) == ""


def test_a_code_later_in_the_prose_is_not_a_heading():
    """Only the opening token counts, or a sentence could invent a collection."""
    assert collection_code("Clicks recorded on BE7A during the survey.") == ""


def test_it_does_not_match_a_bare_word_or_a_bare_number():
    assert collection_code("Squeal.  Reverberation present.") == ""
    assert collection_code("30 ips tape speed.") == ""
    assert collection_code("X  Clicks.") == ""


def test_the_three_codes_that_carry_the_study_are_read_the_same_way():
    """These three cover 3709 of the 4160 kept clips, one species each."""
    assert collection_code("BE7A  Squeal.") == "BE7A"
    assert collection_code("BA2A  Clicks (P. catodon); whistle (Delphinidae).") == "BA2A"
    assert collection_code("AC2A  Whine.") == "AC2A"
