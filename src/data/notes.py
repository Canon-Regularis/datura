"""Reading a Watkins field note.

Every cut carries a written note, for example "Squeal; chirp. Reverberation
present. Good cut." and "Clicks; ship noise." Those notes were never a schema, so
they are read as prose: known terms are looked for and every one found is recorded.
A clip can be several call types at once, and usually is.

Pure text handling. Nothing here reads a file except the vocabulary, and nothing
here touches the network, so the parsing rules can be tested on strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import load_yaml
from src.errors import DaturaError

VOCABULARY_FILE = "configs/call_types.yaml"

CALL_PREFIX = "call_"
CONDITION_PREFIX = "cond_"


class VocabularyError(DaturaError):
    """Raised when the call type vocabulary is missing or malformed."""


@dataclass(frozen=True)
class Vocabulary:
    """Terms to look for in a note, and what to record when they are found.

    Terms are held longest first so that a longer phrase wins: a note saying
    "pulsed call" records a pulsed call, and is not also counted as a call.
    """

    call_types: dict[str, list[str]]
    conditions: dict[str, list[str]]
    max_clip_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def call_labels(self) -> list[str]:
        return list(self.call_types)

    @property
    def condition_labels(self) -> list[str]:
        return list(self.conditions)

    def guard_for(self, call_type: str) -> float | None:
        """The longest clip whose label can still describe a window of it.

        A note is written against a whole cut, not against a moment in it. That is
        harmless for a call that runs throughout a recording, and misleading for one
        that does not: coda labelled clips have a median duration of 64 seconds while
        a coda lasts a few, so most windows of such a clip inherit a label they do
        not deserve. Where a guard is declared, the long clips are left out.
        """
        return self.max_clip_seconds.get(call_type)

    def ordered_terms(self, groups: dict[str, list[str]]) -> list[tuple[str, str]]:
        pairs = [(term, label) for label, terms in groups.items() for term in terms]
        return sorted(pairs, key=lambda pair: -len(pair[0]))


def _parse_group(label: str, entry: object, path: str) -> tuple[list[str], float | None]:
    """Read one vocabulary entry, in either of the two shapes it may take.

    A bare list is just terms. A mapping carries the terms plus whatever else that
    call type needs, which at present is only a clip length guard.
    """
    if isinstance(entry, list):
        return [str(term) for term in entry], None
    if isinstance(entry, dict):
        if "terms" not in entry:
            raise VocabularyError(f"{path}: {label} is a mapping but has no 'terms'")
        unknown = set(entry) - {"terms", "max_clip_seconds"}
        if unknown:
            raise VocabularyError(f"{path}: {label} has unknown keys {sorted(unknown)}")
        guard = entry.get("max_clip_seconds")
        return [str(term) for term in entry["terms"]], None if guard is None else float(guard)
    raise VocabularyError(f"{path}: {label} must be a list of terms or a mapping")


def load_vocabulary(path: str = VOCABULARY_FILE) -> Vocabulary:
    raw = load_yaml(path)
    missing = {"call_types", "conditions"} - set(raw)
    if missing:
        raise VocabularyError(f"{path} is missing sections: {sorted(missing)}")

    call_types: dict[str, list[str]] = {}
    guards: dict[str, float] = {}
    for label, entry in raw["call_types"].items():
        terms, guard = _parse_group(label, entry, path)
        call_types[label] = terms
        if guard is not None:
            guards[label] = guard

    conditions = {
        label: _parse_group(label, entry, path)[0] for label, entry in raw["conditions"].items()
    }
    return Vocabulary(call_types=call_types, conditions=conditions, max_clip_seconds=guards)


def tag_note(note: str | None, vocabulary: Vocabulary) -> tuple[set[str], set[str]]:
    """Every call type and condition mentioned in one note.

    Matched spans are blanked as they are consumed, which is what stops a longer
    phrase from being counted twice under a shorter one.
    """
    if not note:
        return set(), set()

    remaining = note.lower()
    found: dict[str, set[str]] = {"call": set(), "condition": set()}
    for kind, groups in (("call", vocabulary.call_types), ("condition", vocabulary.conditions)):
        for term, label in vocabulary.ordered_terms(groups):
            pattern = re.compile(re.escape(term.lower()))
            if pattern.search(remaining):
                found[kind].add(label)
                remaining = pattern.sub(" " * len(term), remaining)
    return found["call"], found["condition"]


def first_of(value: object) -> object:
    """The first element of a sequence, whatever kind of sequence it is.

    Arrow nests these fields as lists, and ``to_pandas`` hands them back as numpy
    arrays rather than lists. Testing for ``list`` alone therefore matched nothing
    and silently emptied every site in the collection.
    """
    if value is None or isinstance(value, str | bytes | dict):
        return value
    try:
        return next(iter(value), None)
    except TypeError:
        return value


def site_of(location: object) -> str:
    """The named place a recording was made, or an empty string."""
    if not isinstance(location, dict):
        return ""
    name = first_of(location.get("name"))
    return str(name) if name else ""


def coordinates_of(location: object) -> tuple[float | None, float | None]:
    if not isinstance(location, dict):
        return None, None
    point = first_of(location.get("coordinates"))
    if isinstance(point, dict):
        return point.get("lat"), point.get("lon")
    return None, None
