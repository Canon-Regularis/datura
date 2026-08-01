"""Reading a clip's identity out of its path.

Watkins encodes everything you need to group recordings in the filename, so this
is pure string work with no file access. Both the manifest and the audit tables
depend on it, which is why it sits on its own rather than inside either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from src.errors import DaturaError


class ClipPathError(DaturaError):
    """Raised when a path does not look like a Watkins clip."""


@dataclass(frozen=True)
class ClipIdentity:
    """Everything derivable from a clip's path, with no file access."""

    species: str
    year: int
    clip_id: str
    tape_id: str
    cut_id: str


def parse_relative_path(relative: str | PurePosixPath, tape_id_length: int) -> ClipIdentity:
    """Decode ``<Species>/<Year>/<ClipId>.wav``.

    Clip ids come in two forms, seven digits plus a letter (``5401800A``) and eight
    digits (``54018001``). Both encode the same tape in their leading characters, so
    both resolve to tape ``54018``.
    """
    parts = PurePosixPath(str(relative).replace("\\", "/")).parts
    if len(parts) != 3:
        raise ClipPathError(f"expected <Species>/<Year>/<file>.wav, got {relative!r}")

    species, year_text, filename = parts
    if not filename.lower().endswith(".wav"):
        raise ClipPathError(f"not a wav file: {relative!r}")
    if not year_text.isdigit():
        raise ClipPathError(f"not a numeric year directory in {relative!r}")

    clip_id = filename[: -len(".wav")]
    if len(clip_id) < tape_id_length:
        raise ClipPathError(f"clip id {clip_id!r} is shorter than the tape id length")

    return ClipIdentity(
        species=species,
        year=int(year_text),
        clip_id=clip_id,
        tape_id=clip_id[:tape_id_length],
        cut_id=clip_id[tape_id_length:],
    )
