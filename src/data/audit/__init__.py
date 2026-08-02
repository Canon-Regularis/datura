"""Tables that describe the collection, and decide whether to trust it.

Split by what the tables are about: ``collection`` reads the manifest and the
archive, ``context`` reads the parsed field notes. Callers import from here.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import Config
from src.data import annotations
from src.data.audit.collection import audit_tables, cross_species_tapes
from src.data.audit.context import call_types_by_species, site_giveaway, sites_by_species

logger = logging.getLogger(__name__)

__all__ = [
    "annotation_tables",
    "audit_tables",
    "call_types_by_species",
    "cross_species_tapes",
    "log_summary",
    "site_giveaway",
    "sites_by_species",
]


def annotation_tables(cfg: Config, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The site and call type tables, when the field notes have been parsed.

    These are optional: the manifest is complete without them, so a run that has not
    fetched the annotations yet still produces every table it can.
    """
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
