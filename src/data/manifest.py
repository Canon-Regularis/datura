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

import numpy as np
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
    """Add the site and the collection the field notes record, and the groups built on them.

    Folds group on a column of the manifest, and for a long time the only grouping this
    project could express was the tape, because the filename is the only identity a clip
    carries on disk. A tape boundary proves no recording sits on both sides of a fold. A
    place boundary asks the harder question, whether the model survives a recording
    context it has never heard, and answering it needs three columns rather than one.

    ``context`` merges the sites a tape links, ``place`` merges the ones the notes spell
    differently, and ``place_shuffled`` is the control that says how much of a place held
    out score is the place rather than the coarser split. Every one of them is derived
    here so a rebuilt manifest carries them, which a configuration grouping on one of
    them depends on.

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
    joined["context"] = recording_contexts(joined)
    joined["place"] = recording_places(joined)
    joined["place_shuffled"] = shuffled_places(joined, cfg.split.seed)
    return joined


def shuffled_places(manifest: pd.DataFrame, seed: int) -> pd.Series:
    """Pseudo places matching the real ones in every way except which tapes sit together.

    The control for the place held out experiment, and without it that experiment cannot
    be read at all. Grouping by place leaves 24 groups where the tape rule leaves 134, so
    folds are far coarser and every one of them gives up a large share of some class. A
    score that falls under those folds has two explanations, and only one of them is a
    finding.

    So this rebuilds the same split with the geography destroyed. Each real place
    contributes its tape count per species, and the tapes are dealt at random from that
    species' pool, which reproduces the group count, the species mix of every group and
    the tapes per group exactly. A place spanning two species becomes one pseudo group
    fed from both pools, so even that is preserved.

    What it does not reproduce is clips per group, because tapes carry different numbers
    of cuts: the real median is 57 and this deals 44. The control is therefore close
    rather than exact, and the residual favours it.
    """
    kept = manifest[manifest["keep"] & (manifest["place"] != "")]
    if kept.empty:
        return pd.Series("", index=manifest.index, dtype=str)

    # Largest place first, so the dealing is stable rather than dependent on row order.
    shape = kept.groupby(["place", "species"])["tape_id"].nunique().unstack(fill_value=0)
    shape = shape.loc[shape.sum(axis=1).sort_values(ascending=False).index]

    generator = np.random.default_rng(seed)
    pools, taken = {}, dict.fromkeys(shape.columns, 0)
    for species in shape.columns:
        tapes = list(kept.loc[kept["species"] == species, "tape_id"].unique())
        generator.shuffle(tapes)
        pools[species] = tapes

    assignment: dict[str, str] = {}
    for index, (_, row) in enumerate(shape.iterrows()):
        for species, count in row.items():
            if not count:
                continue
            for tape in pools[species][taken[species] : taken[species] + int(count)]:
                assignment[tape] = f"pseudo_{index:02d}"
            taken[species] += int(count)

    for species, used in taken.items():
        if used != len(pools[species]):
            raise ManifestError(
                f"{species}: dealt {used} of {len(pools[species])} tapes into pseudo places"
            )
    return manifest["tape_id"].map(assignment).fillna("")


def recording_places(manifest: pd.DataFrame) -> pd.Series:
    """The coarser group: one physical place, however the notes spelled it.

    ``recording_contexts`` can only merge sites a tape links, so it leaves 39 groups
    covering 26 places and puts more than half the held out clips in a fold whose place
    is also in training. That makes a context held out score optimistic by an amount
    nobody has measured, which is the wrong direction for a leakage test to be wrong in.

    Built on the context rather than beside it, so a merge here can never split a tape:
    contexts are already tape clean, and this only ever joins whole contexts together.
    """
    contexts = manifest["context"].fillna("").astype(str)

    def canonical(value: str) -> str:
        if not value:
            return ""
        lowered = value.lower()
        for alias in PLACE_ALIASES:
            if alias in lowered:
                return alias
        return value

    return contexts.map(canonical)


def recording_contexts(manifest: pd.DataFrame) -> pd.Series:
    """A group that holding out separates by site and by tape at once.

    Grouping on the site alone is not safe here, and measuring it is the only reason
    this exists. Six tapes carry clips the notes place at more than one site, so a
    site boundary put one to three whole tapes on both sides of every fold. Cuts from
    one tape are near duplicates, so those folds were reporting memorisation of a
    recording under the name of a generalisation test.

    Tapes and sites are treated as one graph and a context is a connected component
    of it. A tape linking two sites merges them, which makes the result both site
    clean and tape clean. It also repairs the corpus: `Bermuda` and `Off Gibbs Hill
    Light, Bermuda` are one place written two ways, and the tape that spans them says
    so without anybody hand editing the field notes.

    47 sites become 39 contexts, the scarcest species keeps 10 of them, and no tape
    spans two.
    """
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def root(node: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    pairs = manifest.loc[manifest["site"] != "", ["tape_id", "site"]].drop_duplicates()
    for tape, site in pairs.itertuples(index=False):
        left, right = root(("tape", tape)), root(("site", site))
        if left != right:
            parent[left] = right

    # Named after the first of its sites in sorted order, so the column reads as a
    # place rather than an opaque id, and so the same corpus always names it the same.
    members: dict[tuple[str, str], list[str]] = {}
    for kind, value in parent:
        if kind == "site":
            members.setdefault(root(("site", value)), []).append(value)
    names = {site: sorted(group)[0] for group in members.values() for site in group}

    return manifest["site"].map(lambda site: names.get(site, ""))


# Places the field notes write more than one way. Watkins was transcribed by hand over
# decades, so one anchorage appears as `Dominica`, `DOMINICA`, `N end of Dominica`,
# `3-4 mi. off Roseau, Dominica`, `Dominica Is` and `Dominica IsBD`. The tape graph in
# `recording_contexts` merges any of these that happen to share a tape and cannot merge
# the rest, which leaves one physical place split across several groups and sitting on
# both sides of a fold. Matched as a lowercase substring of the site, first hit wins.
#
# Only names that are unambiguously the same place are listed. `Kikvika, Norway` and
# `Andenes, Norway` are both Norway and are hundreds of kilometres apart, so they stay
# separate: merging on the country would coarsen the split for no gain in honesty.
PLACE_ALIASES: tuple[str, ...] = (
    "dominica",
    "bermuda",
    "oregon",
    "sable",
    "tortola",
    "selina",
    "vancouver",
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
