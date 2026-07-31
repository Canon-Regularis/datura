from __future__ import annotations

import pytest

from src.data.manifest import ManifestError, parse_relative_path


def test_letter_suffixed_and_digit_suffixed_ids_share_a_tape():
    """The two id forms in the collection must resolve to the same recording."""
    lettered = parse_relative_path("HumpbackWhale/1954/5401800A.wav", 5)
    numbered = parse_relative_path("HumpbackWhale/1954/54018001.wav", 5)

    assert lettered.tape_id == numbered.tape_id == "54018"
    assert lettered.cut_id == "00A"
    assert numbered.cut_id == "001"


def test_parses_species_and_year():
    identity = parse_relative_path("SpermWhale/1975/7500501S.wav", 5)

    assert identity.species == "SpermWhale"
    assert identity.year == 1975
    assert identity.clip_id == "7500501S"
    assert identity.tape_id == "75005"


def test_accepts_windows_separators():
    identity = parse_relative_path("KillerWhale\\1964\\6403000A.wav", 5)
    assert identity.tape_id == "64030"


@pytest.mark.parametrize(
    "relative",
    [
        "KillerWhale/6403000A.wav",
        "KillerWhale/1964/extra/6403000A.wav",
        "KillerWhale/1964/6403000A.flac",
        "KillerWhale/nineteen/6403000A.wav",
    ],
)
def test_rejects_malformed_paths(relative):
    with pytest.raises(ManifestError):
        parse_relative_path(relative, 5)


def test_rejects_clip_id_shorter_than_tape_key():
    with pytest.raises(ManifestError):
        parse_relative_path("KillerWhale/1964/640.wav", 5)
