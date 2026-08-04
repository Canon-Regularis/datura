"""What the field notes say about the circumstances of a recording.

Where it was made, what was heard, and what else was going on. These tables exist
because the first phase found that recording bandwidth alone carried most of the
species label; site is the same shape of problem, and this is where it gets
measured rather than assumed.
"""

from __future__ import annotations

import pandas as pd

from src.data.annotations import call_columns
from src.data.notes import CALL_PREFIX


def site_giveaway(manifest: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """How much of the species label the recording site hands over on its own.

    Native sample rate turned out to carry most of what the metadata control knew.
    Location is the same shape of problem: a site visited for one species tells you
    the species before any audio is heard. This table measures that directly, so the
    call type work can be built to avoid repeating the mistake.
    """
    joined = manifest.merge(annotations[["clip_id", "site"]], on="clip_id", how="left")
    species_per_site = joined.groupby("site")["species"].nunique()
    joined["site_species_count"] = joined["site"].map(species_per_site)

    unique_sites = int((species_per_site == 1).sum())
    unique_clips = int((joined["site_species_count"] == 1).sum())
    return pd.DataFrame(
        [
            {
                "sites": len(species_per_site),
                "sites_used_by_one_species": unique_sites,
                "clips": len(joined),
                "clips_at_a_single_species_site": unique_clips,
                "share_of_clips_given_away": round(unique_clips / max(len(joined), 1), 4),
            }
        ]
    )


def code_giveaway(manifest: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """How much of the species label the collection code hands over on its own.

    The same measurement as ``site_giveaway`` against a field nobody had looked at.
    Every note opens with the code of the collection the cut came from, and it turns
    out to separate the species better than either the site or the audio does.
    """
    joined = manifest.merge(annotations[["clip_id", "collection_code"]], on="clip_id", how="left")
    coded = joined[joined["collection_code"].fillna("") != ""]

    species_per_code = coded.groupby("collection_code")["species"].nunique()
    unique_codes = species_per_code[species_per_code == 1].index
    unique_clips = int(coded["collection_code"].isin(unique_codes).sum())

    return pd.DataFrame(
        [
            {
                "codes": len(species_per_code),
                "codes_used_by_one_species": int((species_per_code == 1).sum()),
                "clips": len(joined),
                "clips_carrying_a_code": len(coded),
                "clips_with_a_single_species_code": unique_clips,
                "share_of_clips_given_away": round(unique_clips / max(len(joined), 1), 4),
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
    joined = manifest.merge(annotations[["clip_id", "collection_code"]], on="clip_id", how="left")
    coded = joined[joined["collection_code"].fillna("") != ""]

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
    """Clips and tapes per site per species, largest sites first."""
    joined = manifest.merge(annotations[["clip_id", "site"]], on="clip_id", how="left")
    return (
        joined.groupby(["species", "site"])
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
