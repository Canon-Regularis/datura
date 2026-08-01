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
