"""Build the clip manifest.

The manifest is the one description of what data exists and what is usable. Every
later stage reads it instead of walking the filesystem, so the exclusion rules are
applied in exactly one place.

Identity parsing lives in ``src.data.clips`` and the audit tables in
``src.data.audit``; this module lists the clips and decides which ones are usable.

Usage:
    python -m src.data.manifest [--config configs/base.yaml]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src import cli
from src.audio.io import probe
from src.config import Config
from src.data import annotations, audit
from src.data.clips import parse_relative_path
from src.errors import DaturaError

logger = logging.getLogger(__name__)


class ManifestError(DaturaError):
    """Raised when the extracted tree does not look like the Watkins layout."""


def with_context(cfg: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    """Add the site and the collection the field notes record for each clip.

    Folds group on a column of the manifest, and until now the only grouping this
    project could express was the tape, because the filename is the only identity a
    clip carries on disk. Holding out a site is a different and harder question:
    a tape boundary proves no recording sits on both sides of a fold, and a site
    boundary asks whether the model survives a recording context it has never heard.

    Empty rather than absent where the notes carry nothing, so a caller can tell a
    blank apart from a recorded value. Nothing is dropped here. The manifest says what
    exists and a configuration decides what is usable.
    """
    try:
        parsed = annotations.load(cfg)
    except annotations.AnnotationError:
        logger.info("no parsed field notes yet; the manifest carries no site or collection")
        return manifest

    joined = manifest.merge(
        parsed[["clip_id", "site", "collection_code"]], on="clip_id", how="left"
    )
    for column in ("site", "collection_code"):
        joined[column] = joined[column].fillna("").astype(str)
    return joined


def _drop_reason(header_rate: int, duration: float, cfg: Config) -> str:
    if header_rate < cfg.audio.min_native_sample_rate:
        return "native_rate_below_target"
    if duration < cfg.audio.min_clip_seconds:
        return "clip_too_short"
    return ""


def build_manifest(cfg: Config) -> pd.DataFrame:
    """One row per clip, with the exclusion decision already made."""
    root = cfg.paths.raw / cfg.dataset.archive_root
    if not root.exists():
        raise ManifestError(f"dataset root {root} not found; run python -m src.data.download first")

    rows = []
    for species in cfg.dataset.species:
        species_dir = root / species
        if not species_dir.exists():
            raise ManifestError(f"species directory missing: {species_dir}")
        for path in sorted(species_dir.rglob("*.wav")):
            relative = path.relative_to(root).as_posix()
            identity = parse_relative_path(relative, cfg.split.tape_id_length)
            header = probe(path)
            reason = _drop_reason(header.sample_rate, header.duration_seconds, cfg)
            rows.append(
                {
                    "relative_path": relative,
                    "species": identity.species,
                    "label": cfg.dataset.label_to_index[identity.species],
                    "year": identity.year,
                    "clip_id": identity.clip_id,
                    "tape_id": identity.tape_id,
                    "cut_id": identity.cut_id,
                    "native_sample_rate": header.sample_rate,
                    "channels": header.channels,
                    "frames": header.frames,
                    "subtype": header.subtype,
                    "duration_seconds": header.duration_seconds,
                    "bytes_on_disk": header.bytes_on_disk,
                    "drop_reason": reason,
                    "keep": reason == "",
                }
            )

    if not rows:
        raise ManifestError(f"no wav files found under {root}")
    return pd.DataFrame(rows).sort_values(["species", "clip_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = cli.parser_for(__doc__).parse_args(argv)
    cfg = cli.prepare(args)

    manifest = build_manifest(cfg)
    tables = audit.audit_tables(manifest)
    tables["audit_cross_species_tapes"] = audit.cross_species_tapes(cfg, manifest)
    tables.update(audit.annotation_tables(cfg, manifest))

    # After the audits, which do their own join and would see a duplicated column.
    manifest = with_context(cfg, manifest)

    destination = cfg.paths.metadata / f"manifest_{cfg.corpus}.parquet"
    manifest.to_parquet(destination, index=False)
    for stem, table in tables.items():
        table.to_csv(cfg.paths.metadata / f"{stem}_{cfg.corpus}.csv", index=False)

    audit.log_summary(manifest, tables)
    logger.info("\nmanifest written to %s", destination)
    return 0


def manifest_path(cfg: Config) -> Path:
    return cfg.paths.metadata / f"manifest_{cfg.corpus}.parquet"


def load_manifest(cfg: Config, *, kept_only: bool = True) -> pd.DataFrame:
    path = manifest_path(cfg)
    if not path.exists():
        raise ManifestError(f"{path} not found; run python -m src.data.manifest first")
    frame = pd.read_parquet(path)
    return frame[frame["keep"]].reset_index(drop=True) if kept_only else frame


if __name__ == "__main__":
    sys.exit(main())
