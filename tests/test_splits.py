"""The most important tests in the project.

Cuts from one tape are near duplicates. If a tape straddles a fold boundary the
reported score measures memorisation of that recording, so these tests exist to
make that failure impossible to introduce quietly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splits import (
    SplitError,
    assert_no_tape_leak,
    clips_from_index,
    fold_summary,
    make_folds,
    rows_for_clips,
    tape_id_of,
)

SPECIES = ["HumpbackWhale", "SpermWhale", "KillerWhale"]


def build_clips(tapes_per_species: int = 8, cuts_per_tape: int = 4) -> pd.DataFrame:
    rows = []
    for label, species in enumerate(SPECIES):
        for tape in range(tapes_per_species):
            tape_id = f"{label}{tape:04d}"
            for cut in range(cuts_per_tape):
                rows.append(
                    {
                        "clip_id": f"{tape_id}{cut:03d}",
                        "tape_id": tape_id,
                        "species": species,
                        "label": label,
                    }
                )
    return pd.DataFrame(rows)


def test_tape_id_collapses_both_clip_id_forms():
    assert tape_id_of("5401800A", 5) == tape_id_of("54018001", 5) == "54018"


def test_no_tape_crosses_a_fold_boundary(config):
    clips = build_clips()
    folds = make_folds(clips, config)

    for fold in folds:
        train = {tape_id_of(c, 5) for c in fold.train_clips}
        validation = {tape_id_of(c, 5) for c in fold.validation_clips}
        test = {tape_id_of(c, 5) for c in fold.test_clips}
        assert not train & test
        assert not validation & test
        assert not train & validation


def test_folds_partition_every_clip(config):
    clips = build_clips()
    folds = make_folds(clips, config)

    for fold in folds:
        covered = set(fold.train_clips) | set(fold.validation_clips) | set(fold.test_clips)
        assert covered == set(clips["clip_id"])

    test_clips = [clip for fold in folds for clip in fold.test_clips]
    assert sorted(test_clips) == sorted(clips["clip_id"])


def test_every_fold_sees_all_three_species(config):
    clips = build_clips()
    folds = make_folds(clips, config)
    by_clip = clips.set_index("clip_id")

    for fold in folds:
        for part in (fold.train_clips, fold.test_clips):
            assert set(by_clip.loc[list(part), "species"]) == set(SPECIES)


def test_folds_are_reproducible_from_the_seed(config):
    clips = build_clips()
    first = make_folds(clips, config)
    second = make_folds(clips, config)

    assert [f.test_clips for f in first] == [f.test_clips for f in second]
    assert [f.validation_clips for f in first] == [f.validation_clips for f in second]


def test_leak_check_rejects_a_shared_tape(config):
    clips = build_clips()
    folds = make_folds(clips, config)
    leaky = folds[0]
    tampered = type(leaky)(
        index=leaky.index,
        train_clips=(*leaky.train_clips, leaky.test_clips[0]),
        validation_clips=leaky.validation_clips,
        test_clips=leaky.test_clips,
    )

    with pytest.raises(SplitError, match="leaks"):
        assert_no_tape_leak(clips, [tampered], config.split.tape_id_length)


def test_rejects_duplicate_clip_ids(config):
    clips = pd.concat([build_clips(), build_clips().head(1)], ignore_index=True)
    with pytest.raises(SplitError, match="unique"):
        make_folds(clips, config)


def test_rejects_more_folds_than_tapes(config):
    clips = build_clips(tapes_per_species=1, cuts_per_tape=6)
    with pytest.raises(SplitError, match="cannot fill"):
        make_folds(clips, config)


def test_window_rows_follow_their_clip(config):
    clips = build_clips()
    index = clips.loc[clips.index.repeat(3)].reset_index(drop=True)
    folds = make_folds(clips, config)

    train_rows = rows_for_clips(index, folds[0].train_clips)
    test_rows = rows_for_clips(index, folds[0].test_clips)

    assert len(set(train_rows) & set(test_rows)) == 0
    assert len(train_rows) == 3 * len(folds[0].train_clips)
    assert set(index.iloc[train_rows]["clip_id"]) == set(folds[0].train_clips)


def test_clips_from_index_collapses_windows():
    clips = build_clips()
    index = clips.loc[clips.index.repeat(5)].reset_index(drop=True)
    collapsed = clips_from_index(index)

    assert len(collapsed) == len(clips)
    assert set(collapsed.columns) == {"clip_id", "tape_id", "species", "label"}


def test_fold_summary_reports_tapes_per_part(config):
    clips = build_clips()
    folds = make_folds(clips, config)
    summary = fold_summary(clips, folds)

    assert set(summary["part"]) == {"train", "validation", "test"}
    assert summary["tapes"].min() >= 1
    assert len(summary) == len(folds) * 3 * len(SPECIES)
