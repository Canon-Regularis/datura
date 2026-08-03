"""Reading the call type vocabulary off disk.

An entry may be a bare list of terms or a mapping carrying those terms plus a clip
length guard. Both shapes have to produce the same labels, because the guard says
nothing about what a note means: it only says which clips are long enough that
their label stops describing a window of them.
"""

from __future__ import annotations

import pytest
import yaml

from src.data.notes import VocabularyError, load_vocabulary, tag_note

BARE = {
    "call_types": {"click": ["click"], "coda": ["coda"]},
    "conditions": {"ship_noise": ["ship noise", "propeller"]},
}

WITH_GUARD = {
    "call_types": {
        "click": ["click"],
        "coda": {"terms": ["coda"], "max_clip_seconds": 8},
    },
    "conditions": {"ship_noise": ["ship noise", "propeller"]},
}


def write(tmp_path, raw: dict, name: str = "call_types") -> str:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_both_shapes_produce_the_same_labels(tmp_path):
    bare = load_vocabulary(write(tmp_path, BARE, "bare"))
    guarded = load_vocabulary(write(tmp_path, WITH_GUARD, "guarded"))

    assert bare.call_types == guarded.call_types
    assert bare.conditions == guarded.conditions

    note = "Codas; clicks.  Ship noise."
    assert tag_note(note, bare) == tag_note(note, guarded)


def test_a_bare_list_declares_no_guard(tmp_path):
    vocabulary = load_vocabulary(write(tmp_path, BARE))

    assert vocabulary.max_clip_seconds == {}
    assert vocabulary.guard_for("coda") is None


def test_a_guard_is_read_for_the_call_type_that_declares_it(tmp_path):
    vocabulary = load_vocabulary(write(tmp_path, WITH_GUARD))

    assert vocabulary.guard_for("coda") == 8.0
    assert vocabulary.guard_for("click") is None


def test_a_mapping_without_terms_is_refused(tmp_path):
    raw = {"call_types": {"coda": {"max_clip_seconds": 8}}, "conditions": {}}

    with pytest.raises(VocabularyError, match="terms"):
        load_vocabulary(write(tmp_path, raw))


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path):
    raw = {"call_types": {"coda": {"terms": ["coda"], "max_clip_secs": 8}}, "conditions": {}}

    with pytest.raises(VocabularyError, match="max_clip_secs"):
        load_vocabulary(write(tmp_path, raw))


def test_the_shipped_vocabulary_guards_only_the_episodic_calls():
    vocabulary = load_vocabulary()

    assert vocabulary.guard_for("coda") == 8.0
    assert vocabulary.guard_for("click") is None, (
        "clicks run throughout a cut, so they need no guard"
    )
