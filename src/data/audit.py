"""Tables that describe the collection, and decide whether to trust it.

The manifest says what exists. These tables say what that means: which sample
rates each species was recorded at, how many independent tapes are really behind a
clip count, and which tapes carry more than one species. They are the evidence
behind every design decision downstream, so they are built once and written beside
the manifest.
"""

from __future__ import annotations

import logging
import zipfile

import pandas as pd

from src.config import Config
from src.data.clips import ClipPathError, parse_relative_path

logger = logging.getLogger(__name__)


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
    from src.data.annotations import CALL_PREFIX, call_columns

    columns = call_columns(annotations)
    joined = manifest.merge(annotations[["clip_id", *columns]], on="clip_id", how="left")
    counts = joined.groupby("species")[columns].sum().T.astype(int)
    counts.index = [name.removeprefix(CALL_PREFIX) for name in counts.index]
    counts.index.name = "call_type"
    counts["total"] = counts.sum(axis=1)
    counts["species_with_any"] = (counts.drop(columns="total") > 0).sum(axis=1)
    return counts.reset_index().sort_values("total", ascending=False)


def annotation_tables(cfg: Config, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The site and call type tables, when the field notes have been parsed.

    These are optional: the manifest is complete without them, so a run that has
    not fetched the annotations yet still produces every table it can.
    """
    from src.data import annotations

    try:
        parsed = annotations.load(cfg)
    except annotations.AnnotationError:
        logger.info(
            "no parsed field notes yet; run python -m src.data.annotations for the "
            "site and call type tables"
        )
        return {}

    kept = manifest[manifest["keep"]]
    return {
        "audit_site_giveaway": site_giveaway(kept, parsed),
        "audit_sites_by_species": sites_by_species(kept, parsed),
        "audit_call_types": call_types_by_species(kept, parsed),
    }


def cross_species_tapes(cfg: Config) -> pd.DataFrame:
    """Tapes carrying more than one species label, read from the zip's index.

    This covers all 54 species rather than the three under study, because a tape
    shared with an unused species still tells you how the collection was cut. The
    zip is only indexed here, never decompressed.
    """
    archive = cfg.paths.raw / cfg.dataset.zip_name
    if not archive.exists():
        return pd.DataFrame(columns=["tape_id", "n_species", "species"])

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
        {"tape_id": tape, "n_species": len(names), "species": ", ".join(sorted(names))}
        for tape, names in sorted(tapes.items())
        if len(names) > 1
    ]
    return pd.DataFrame(shared, columns=["tape_id", "n_species", "species"])


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


def log_summary(manifest: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    logger.info("\nCoverage before and after the sample rate filter")
    logger.info(tables["audit_coverage"].to_string(index=False))
    if not tables["audit_dropped"].empty:
        logger.info("\nDropped clips")
        logger.info(tables["audit_dropped"].to_string(index=False))
    logger.info("\nNative sample rates present")
    logger.info(tables["audit_sample_rates"].to_string(index=False))
    kept = manifest["keep"].sum()
    logger.info(
        "\n%d of %d clips kept across %d species",
        kept,
        len(manifest),
        manifest["species"].nunique(),
    )
