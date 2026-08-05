"""What the manifest itself says about the collection.

Coverage before and after the sample rate filter, which rates each species was
recorded at, how many independent tapes sit behind a clip count, and which tapes
carry more than one species. These are the tables that decide whether any
downstream number is trustworthy.
"""

from __future__ import annotations

import zipfile

import pandas as pd

from src.config import Config
from src.data.clips import ClipPathError, parse_relative_path


def cross_species_tapes(cfg: Config) -> pd.DataFrame:
    """Tapes carrying more than one species label, read from the zip's index.

    Every row covers all 54 species, because a tape shared with an unused species
    still says something about how the collection was cut. The last two columns then
    narrow it to the configuration in hand, which is the part that bears on a per
    class score: a tape carrying two species under study is a group with two labels,
    kept whole so nothing leaks, but pooled in every recall it contributes to.

    Without those columns this table was byte identical for every configuration, so
    the file named after the wide set could not answer the question the wide set
    raises. The zip is only indexed here, never decompressed.
    """
    studied = set(cfg.dataset.species)
    archive = cfg.paths.raw / cfg.dataset.zip_name
    if not archive.exists():
        return pd.DataFrame(
            columns=["tape_id", "n_species", "species", "n_under_study", "under_study"]
        )

    prefix = f"{cfg.dataset.archive_root}/"
    tapes: dict[str, set[str]] = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not member.startswith(prefix) or not member.lower().endswith(".wav"):
                continue
            try:
                identity = parse_relative_path(member[len(prefix) :], cfg.split.tape_id_length)
            except ClipPathError:
                continue
            tapes.setdefault(identity.tape_id, set()).add(identity.species)

    shared = [
        {
            "tape_id": tape,
            "n_species": len(names),
            "species": ", ".join(sorted(names)),
            "n_under_study": len(names & studied),
            "under_study": ", ".join(sorted(names & studied)),
        }
        for tape, names in sorted(tapes.items())
        if len(names) > 1
    ]
    columns = ["tape_id", "n_species", "species", "n_under_study", "under_study"]
    return pd.DataFrame(shared, columns=columns)


def audit_tables(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The tables that decide whether any downstream number is trustworthy."""
    kept = manifest[manifest["keep"]]

    coverage = (
        manifest.groupby("species")
        .agg(clips=("clip_id", "size"), tapes=("tape_id", "nunique"))
        .join(
            kept.groupby("species").agg(
                kept_clips=("clip_id", "size"),
                kept_tapes=("tape_id", "nunique"),
                kept_hours=("duration_seconds", lambda s: s.sum() / 3600.0),
            )
        )
        .fillna(0)
        .astype({"kept_clips": int, "kept_tapes": int})
        .reset_index()
    )

    rates = (
        manifest.groupby(["species", "native_sample_rate"])
        .agg(clips=("clip_id", "size"), tapes=("tape_id", "nunique"))
        .reset_index()
        .sort_values(["species", "native_sample_rate"])
    )

    per_tape = (
        manifest.groupby(["species", "tape_id"])
        .agg(
            clips=("clip_id", "size"),
            kept_clips=("keep", "sum"),
            year=("year", "first"),
            native_sample_rate=("native_sample_rate", "first"),
            seconds=("duration_seconds", "sum"),
        )
        .reset_index()
        .sort_values(["species", "tape_id"])
    )

    dropped = (
        manifest[~manifest["keep"]]
        .groupby(["species", "drop_reason"])
        .agg(clips=("clip_id", "size"), tapes=("tape_id", "nunique"))
        .reset_index()
    )

    return {
        "audit_coverage": coverage,
        "audit_sample_rates": rates,
        "audit_per_tape": per_tape,
        "audit_dropped": dropped,
    }
