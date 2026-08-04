"""Whether the collection code is a confound or an artefact of how it was joined.

A field that identifies the species better than the audio does is either the most
important thing in this dataset or a bug in a merge. The difference is whether it
survives the fold boundary, and that turns on one fact: a code spans many tapes, so
a held out tape almost always carries a code the training tapes carried too.

The falsifier is here. Put every code on its own tape and the advantage has to
vanish, because then tape grouped folds already handle it. If it does not vanish,
the label is leaking through the join and nothing else in this file matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.splits import folds_for_index, rows_for_clips
from src.features.controls import LogbookFeatureSource, MetadataFeatureSource

SPECIES = ["HumpbackWhale", "SpermWhale", "KillerWhale"]


def collection(tapes_per_species: int, clips_per_tape: int, *, code_per_tape: bool) -> pd.DataFrame:
    """Windows over some tapes, with a collection code laid out one way or the other.

    ``code_per_tape`` gives every tape its own code, which is a tape id wearing a
    hat. Otherwise one code covers a whole species, which is what the real notes do.
    """
    rows = []
    for label, species in enumerate(SPECIES):
        for tape in range(tapes_per_species):
            tape_id = f"{label}{tape:04d}"
            code = f"C{label}{tape:02d}" if code_per_tape else f"C{label}"
            for clip in range(clips_per_tape):
                rows.append(
                    {
                        "clip_id": f"{tape_id}{clip:03d}",
                        "tape_id": tape_id,
                        "species": species,
                        "label": label,
                        "collection_code": code,
                        # Deliberately uninformative, so the code is the only signal.
                        "site": "one place",
                        "latitude": 0.0,
                        "longitude": 0.0,
                        "native_sample_rate": 10000,
                        "year": 1970,
                        "duration_seconds": 2.0,
                        "bytes_on_disk": 1000,
                    }
                )
    return pd.DataFrame(rows)


def score_across_folds(index: pd.DataFrame, source, cfg) -> float:
    """Mean accuracy of a nearest rule fitted per fold on the training tapes.

    A decision tree would do here, but a lookup keeps the test free of a dependency
    on how xgboost splits and makes the mechanism obvious: whatever a training tape
    says about a feature value is what a test tape gets.
    """
    matrix = source.matrix(np.arange(len(index))).to_numpy()
    scores = []
    for fold in folds_for_index(index, cfg):
        train = rows_for_clips(index, fold.train_clips)
        test = rows_for_clips(index, fold.test_clips)

        table: dict[tuple, int] = {}
        for row in train:
            table.setdefault(tuple(matrix[row]), index["label"].iloc[row])
        majority = int(index["label"].iloc[train].mode().iloc[0])

        predicted = [table.get(tuple(matrix[row]), majority) for row in test]
        scores.append(float(np.mean(np.array(predicted) == index["label"].iloc[test].to_numpy())))
    return float(np.mean(scores))


def test_a_code_spanning_many_tapes_crosses_the_fold_boundary(config):
    """The real layout: one collection per species, over many tapes."""
    index = collection(9, 4, code_per_tape=False)

    logbook = score_across_folds(index, LogbookFeatureSource(index, []), config)
    metadata = score_across_folds(index, MetadataFeatureSource(index), config)

    assert logbook > 0.95, "a code shared across tapes is visible from the training side"
    assert logbook > metadata + 0.3, "and it is worth far more than the equipment metadata"


def test_a_code_confined_to_one_tape_is_worth_nothing(config):
    """The falsifier. Tape grouped folds already handle a per tape identifier.

    If this ever passes with a high score, the collection code result is a leak in
    the merge rather than a fact about the collection.
    """
    index = collection(9, 4, code_per_tape=True)

    logbook = score_across_folds(index, LogbookFeatureSource(index, []), config)
    metadata = score_across_folds(index, MetadataFeatureSource(index), config)

    assert logbook <= metadata + 0.05, (
        "a code that never leaves its own tape cannot help across a tape grouped fold"
    )


def test_the_real_codes_span_many_tapes_each():
    """The property the whole finding rests on, asserted against the committed audit."""
    import pathlib

    path = pathlib.Path("data/metadata/audit_codes_by_species_base_10k.csv")
    if not path.exists():
        import pytest

        pytest.skip("audit tables absent; run python -m src.data.manifest first")

    table = pd.read_csv(path).sort_values("clips", ascending=False)
    biggest = table.head(3)

    assert (biggest["species"] == 1).all(), "each of the three carries exactly one species"
    assert (biggest["tapes"] >= 10).all(), "and each spans many tapes, so folds cannot contain it"
    assert biggest["clips"].sum() / table["clips"].sum() > 0.9
