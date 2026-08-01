"""Build the clip manifest and the dataset audit tables.

The manifest is the one description of what data exists and what is usable. Every
later stage reads it instead of walking the filesystem, so the exclusion rules are
applied in exactly one place.

Usage:
    python -m src.data.manifest [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

from src.audio.io import probe
from src.config import Config, load_config


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClipIdentity:
    """Everything derivable from a clip's path, with no file access."""

    species: str
    year: int
    clip_id: str
    tape_id: str
    cut_id: str


def parse_relative_path(relative: str | PurePosixPath, tape_id_length: int) -> ClipIdentity:
    """Decode ``<Species>/<Year>/<ClipId>.wav``.

    Clip ids come in two forms, seven digits plus a letter (``5401800A``) and eight
    digits (``54018001``). Both encode the same tape in their leading characters, so
    both resolve to tape ``54018``.
    """
    parts = PurePosixPath(str(relative).replace("\\", "/")).parts
    if len(parts) != 3:
        raise ManifestError(f"expected <Species>/<Year>/<file>.wav, got {relative!r}")

    species, year_text, filename = parts
    if not filename.lower().endswith(".wav"):
        raise ManifestError(f"not a wav file: {relative!r}")
    if not year_text.isdigit():
        raise ManifestError(f"non-numeric year directory in {relative!r}")

    clip_id = filename[: -len(".wav")]
    if len(clip_id) < tape_id_length:
        raise ManifestError(f"clip id {clip_id!r} is shorter than the tape id length")

    return ClipIdentity(
        species=species,
        year=int(year_text),
        clip_id=clip_id,
        tape_id=clip_id[:tape_id_length],
        cut_id=clip_id[tape_id_length:],
    )


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
            except ManifestError:
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


def _print_summary(manifest: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    print("\nCoverage before and after the sample rate filter")
    print(tables["audit_coverage"].to_string(index=False))
    if not tables["audit_dropped"].empty:
        print("\nDropped clips")
        print(tables["audit_dropped"].to_string(index=False))
    print("\nNative sample rates present")
    print(tables["audit_sample_rates"].to_string(index=False))
    kept = manifest["keep"].sum()
    print(f"\n{kept} of {len(manifest)} clips kept across {manifest['species'].nunique()} species")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure()

    manifest = build_manifest(cfg)
    tables = audit_tables(manifest)
    tables["audit_cross_species_tapes"] = cross_species_tapes(cfg)

    destination = cfg.paths.metadata / f"manifest_{cfg.name}.parquet"
    manifest.to_parquet(destination, index=False)
    for stem, table in tables.items():
        table.to_csv(cfg.paths.metadata / f"{stem}_{cfg.name}.csv", index=False)

    _print_summary(manifest, tables)
    print(f"\nmanifest written to {destination}")
    return 0


def manifest_path(cfg: Config) -> Path:
    return cfg.paths.metadata / f"manifest_{cfg.name}.parquet"


def load_manifest(cfg: Config, *, kept_only: bool = True) -> pd.DataFrame:
    path = manifest_path(cfg)
    if not path.exists():
        raise ManifestError(f"{path} not found; run python -m src.data.manifest first")
    frame = pd.read_parquet(path)
    return frame[frame["keep"]].reset_index(drop=True) if kept_only else frame


if __name__ == "__main__":
    sys.exit(main())
