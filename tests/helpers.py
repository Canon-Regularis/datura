"""What more than one test file needs, written once.

Six files built the same clip frame and five spelled out the same species list, so a
change to what a clip row carries meant six correct edits and nothing caught five.

Everything here is a fixture rather than a fact about the project. Facts belong in the
test that asserts them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest

from src.config import PROJECT_ROOT

# The three species every synthetic fixture uses, in label order. Their names matter
# because the controls key on them and the reports order rows by them.
SPECIES: tuple[str, ...] = ("HumpbackWhale", "SpermWhale", "KillerWhale")


def clip_rows(
    tapes_per_species: int = 8,
    clips_per_tape: int = 4,
    *,
    species: Sequence[str] = SPECIES,
    extra: Callable[[dict], dict] | None = None,
) -> pd.DataFrame:
    """One row per clip, over a tidy layout of tapes within species.

    Ids follow the real shape closely enough for the split rules to work on them: the
    first five characters are the tape, which is what ``tape_id_length`` cuts on, and
    the last three are the cut within it.

    ``extra`` is handed each row and returns whatever else that row should carry, so a
    test needing a collection code or a coordinate builds on this rather than beside
    it. It sees the row so it can key on the tape.
    """
    rows = [
        {
            "clip_id": f"{label}{tape:04d}{clip:03d}",
            "tape_id": f"{label}{tape:04d}",
            "species": name,
            "label": label,
        }
        for label, name in enumerate(species)
        for tape in range(tapes_per_species)
        for clip in range(clips_per_tape)
    ]
    if extra is not None:
        rows = [row | extra(row) for row in rows]
    return pd.DataFrame(rows)


def window_rows(windows_per_clip: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Several windows per clip, for the aggregation that turns them back into clips.

    Keyed by clip id rather than generated, because what these tests vary is how many
    windows one clip has against another.
    """
    return pd.DataFrame(
        [
            {
                "clip_id": clip_id,
                "tape_id": clip_id[:5],
                "species": SPECIES[label],
                "label": label,
                "window_index": position,
            }
            for clip_id, (label, count) in windows_per_clip.items()
            for position in range(count)
        ]
    )


def needs(path: Path, hint: str = "run the pipeline") -> Path:
    """The artifact, or a skip that says what would produce it.

    33 tests read something the pipeline writes, and a fresh clone has none of it. They
    each spelled the check and the message themselves, so the reasons ranged from a
    full command to the word "absent". Now they say the same thing and ``-ra`` prints
    it, which is the difference between a green run that tested nothing and one that
    says which four commands it is waiting on.
    """
    if not path.exists():
        try:
            shown = path.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = path
        pytest.skip(f"{shown} absent; {hint}")
    return path


def published() -> str:
    """Every document that quotes a number, as one string.

    The README is the entry point and the longer arguments live under ``docs/``. A
    figure is checked wherever it is printed, so moving a table between the two is a
    question of what a reader wants first rather than of what stays honest.
    """
    parts = [(PROJECT_ROOT / "README.md").read_text(encoding="utf-8")]
    parts += [
        path.read_text(encoding="utf-8") for path in sorted((PROJECT_ROOT / "docs").glob("*.md"))
    ]
    return "\n\n".join(parts)


def prose() -> str:
    """The published text with its line breaks collapsed.

    Every phrase checked against an artifact is one sentence in a document that wraps
    where it suits a reader. Matching the raw text made a rewrap look like a wrong
    number, so the wrapping comes out before the check rather than being pinned.
    """
    return " ".join(published().split())
