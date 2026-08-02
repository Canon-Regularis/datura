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
from dataclasses import dataclass

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

    @property
    def call_labels(self) -> list[str]:
        return list(self.call_types)

    @property
    def condition_labels(self) -> list[str]:
        return list(self.conditions)

    def ordered_terms(self, groups: dict[str, list[str]]) -> list[tuple[str, str]]:
        pairs = [(term, label) for label, terms in groups.items() for term in terms]
        return sorted(pairs, key=lambda pair: -len(pair[0]))


def load_vocabulary(path: str = VOCABULARY_FILE) -> Vocabulary:
    raw = load_yaml(path)
    missing = {"call_types", "conditions"} - set(raw)
    if missing:
        raise VocabularyError(f"{path} is missing sections: {sorted(missing)}")
    return Vocabulary(call_types=raw["call_types"], conditions=raw["conditions"])


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
