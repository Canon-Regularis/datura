"""What the field notes say about the circumstances of a recording.

Where it was made, what was heard, and what else was going on. These tables exist
because the first phase found that recording bandwidth alone carried most of the
species label; site is the same shape of problem, and this is where it gets
measured rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.annotations import call_columns
from src.data.notes import CALL_PREFIX

# The fields a held out score is split on, and what to call each one in a report.
# Native sample rate was the first field found to hand over the species; the
# collection code the field note opens with is the second and the stronger. Site is
# measured by ``site_giveaway`` but not split on, because it is nearly constant
# within a collection and would say the same thing twice.
GIVEAWAY_FIELDS = {
    "native_sample_rate": "native sample rate",
    "collection_code": "collection code",
}


# What one field can do to the species label of a clip.
UNIQUE = "unique"
SHARED = "shared"
ABSENT = "absent"


def with_columns(frame: pd.DataFrame, parsed: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """The frame carrying the named note columns, joining only what is missing.

    The manifest gained ``site`` and ``collection_code`` when folds had to be able to
    group on them. Merging them in again here would give pandas two columns of the
    same name and leave every caller reading ``site_x``, so the join is conditional
    rather than unconditional.
    """
    absent = [column for column in columns if column not in frame.columns]
    if not absent:
        return frame
    return frame.merge(parsed[["clip_id", *absent]], on="clip_id", how="left")


def recorded(frame: pd.DataFrame, column: str) -> pd.Series:
    """Rows where the field was actually written down.

    A blank collection code is not a code that several species happen to share. It is
    the absence of one. Counting the blank as a value put 359 base_10k clips in a
    bucket labelled as though a code had been recorded and found ambiguous, when no
    code in that configuration is used by more than one species at all.
    """
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        return values.notna()
    # Asking whether the dtype is object would miss it. Text columns arrive from
    # parquet as a string dtype, and every blank in one of those is not null, so a
    # null check alone calls the empty string a recorded value.
    return values.fillna("").astype(str).str.strip() != ""


def species_per_value(frame: pd.DataFrame, column: str) -> pd.Series:
    """How many species each recorded value of one column is used by."""
    present = frame[recorded(frame, column)]
    return present.groupby(column)["species"].nunique()


def shared_values(frame: pd.DataFrame, column: str) -> set:
    """The recorded values of one column that more than one species uses.

    A value used by exactly one species names that species. Everything asking how
    much a field gives away is asking about this set, or about its complement, so it
    is computed once.
    """
    counts = species_per_value(frame, column)
    return set(counts[counts > 1].index)


def giveaway_labels(
    manifest: pd.DataFrame, annotations: pd.DataFrame | None = None
) -> dict[str, pd.Series]:
    """Per field, a clip indexed label saying what its value does to the species.

    Three outcomes, because two cannot describe the data. A value used by one species
    names it. A value used by several does not. A clip carrying no value at all is a
    third case, and folding it into either of the others states something false about
    every clip in the bucket.

    Splitting held out clips on this asks the question a headline score cannot: where
    the recording does not name the species by itself, does listening to it still
    help? A field the notes do not carry is left out rather than guessed at, so a
    collection whose annotations were never fetched still gets the sample rate split.
    """
    frame = manifest
    if annotations is not None:
        frame = with_columns(manifest, annotations, "collection_code")

    labels = {}
    for column, label in GIVEAWAY_FIELDS.items():
        if column not in frame.columns:
            continue
        shared = shared_values(frame, column)
        outcome = np.where(
            ~recorded(frame, column),
            ABSENT,
            np.where(frame[column].isin(shared), SHARED, UNIQUE),
        )
        labels[label] = pd.Series(
            outcome, index=pd.Index(frame["clip_id"], name="clip_id"), name=label
        )
    return labels


def site_giveaway(manifest: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """How much of the species label the recording site hands over on its own.

    Native sample rate turned out to carry most of what the metadata control knew.
    Location is the same shape of problem: a site visited for one species tells you
    the species before any audio is heard. This table measures that directly, so the
    call type work can be built to avoid repeating the mistake.

    Clips with no site recorded are counted in ``clips`` and nowhere else. A blank is
    not a site that happens to be visited for several species, and reporting it as one
    put an empty pseudo site in the count: base_10k has 47 sites and used to say 48.
    """
    frame = with_columns(manifest, annotations, "site")
    species_per_site = species_per_value(frame, "site")
    frame = frame.assign(site_species_count=frame["site"].map(species_per_site))

    unique_sites = int((species_per_site == 1).sum())
    unique_clips = int((frame["site_species_count"] == 1).sum())
    return pd.DataFrame(
        [
            {
                "sites": len(species_per_site),
                "sites_used_by_one_species": unique_sites,
                "clips": len(frame),
                "clips_carrying_a_site": int(recorded(frame, "site").sum()),
                "clips_at_a_single_species_site": unique_clips,
                "share_of_clips_given_away": round(unique_clips / max(len(frame), 1), 4),
            }
        ]
    )


def code_giveaway(manifest: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """How much of the species label the collection code hands over on its own.

    The same measurement as ``site_giveaway`` against a field nobody had looked at.
    Every note opens with the code of the collection the cut came from, and it turns
    out to separate the species better than either the site or the audio does.
    """
    frame = with_columns(manifest, annotations, "collection_code")
    coded = frame[recorded(frame, "collection_code")]

    species_per_code = species_per_value(coded, "collection_code")
    unique_codes = species_per_code[species_per_code == 1].index
    unique_clips = int(coded["collection_code"].isin(unique_codes).sum())

    return pd.DataFrame(
        [
            {
                "codes": len(species_per_code),
                "codes_used_by_one_species": int((species_per_code == 1).sum()),
                "clips": len(frame),
                "clips_carrying_a_code": len(coded),
                "clips_with_a_single_species_code": unique_clips,
                "share_of_clips_given_away": round(unique_clips / max(len(frame), 1), 4),
            }
        ]
    )


def codes_by_species(manifest: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Clips, tapes and species behind each collection code.

    The tapes column is the one that matters. A code sitting on a single tape would
    be a tape id under another name, and tape grouped folds would already handle it.
    These span dozens of tapes each, which is why they cross the fold boundary and
    why the control has to see them.
    """
    frame = with_columns(manifest, annotations, "collection_code")
    coded = frame[recorded(frame, "collection_code")]

    table = (
        coded.groupby("collection_code")
        .agg(
            clips=("clip_id", "size"),
            tapes=("tape_id", "nunique"),
            species=("species", "nunique"),
            carried=("species", lambda names: ", ".join(sorted(set(names)))),
        )
        .reset_index()
    )
    return table.sort_values("clips", ascending=False)


def sites_by_species(manifest: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Clips and tapes per site per species, largest sites first.

    Clips with no site are left out, so every row here names a place.
    """
    frame = with_columns(manifest, annotations, "site")
    frame = frame[recorded(frame, "site")]
    return (
        frame.groupby(["species", "site"])
        .agg(clips=("clip_id", "size"), tapes=("tape_id", "nunique"))
        .reset_index()
        .sort_values(["species", "clips"], ascending=[True, False])
    )


def call_types_by_species(manifest: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """How many clips of each species carry each call type.

    Read this before treating call type as a task in its own right. Several types
    sit almost entirely within one species, so a model trained across species would
    be relearning species under another name.
    """

    columns = call_columns(annotations)
    joined = manifest.merge(annotations[["clip_id", *columns]], on="clip_id", how="left")
    counts = joined.groupby("species")[columns].sum().T.astype(int)
    counts.index = [name.removeprefix(CALL_PREFIX) for name in counts.index]
    counts.index.name = "call_type"
    counts["total"] = counts.sum(axis=1)
    counts["species_with_any"] = (counts.drop(columns="total") > 0).sum(axis=1)
    return counts.reset_index().sort_values("total", ascending=False)
