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
