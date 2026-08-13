"""The columns folds are allowed to group on.

A fold proves whatever its grouping column proves and nothing more. Grouping on the
tape says no recording sits on both sides of a split, which is the weakest claim worth
making. Grouping on the place says the model survives a recording context it has never
heard, which is the claim the report is actually about.

Each column here is one grouping, derived from the field notes rather than declared, so
a rebuilt manifest carries every one of them and a configuration only has to name the
one it wants. They form a ladder: ``context`` merges the sites a single tape links,
``place`` merges the ones the notes spell differently, and ``place_shuffled`` rebuilds
the place split with the geography destroyed so the coarseness can be subtracted.

This was inside ``src.data.manifest``, which lists clips and decides which are usable.
Deciding what a fold is allowed to see is a different question with a different reason
to change, and the two only meet at ``with_context``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config
from src.data import annotations as field_notes
from src.errors import DaturaError

logger = logging.getLogger(__name__)

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


class GroupingError(DaturaError):
    """Raised when a derived grouping does not account for every tape it was given."""


def with_context(cfg: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    """Add the site and the collection the field notes record, and the groups built on them.

    Every grouping is derived here so a rebuilt manifest carries all of them, which a
    configuration grouping on one of them depends on.

    Empty rather than absent where the notes carry nothing, so a caller can tell a
    blank apart from a recorded value. Nothing is dropped here. The manifest says what
    exists and a configuration decides what is usable.
    """
    try:
        parsed = field_notes.load(cfg)
    except field_notes.AnnotationError:
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

    A tape carrying two species is dealt from one pool only. Dealing it from both would
    assign it twice and the second assignment would win, which would break the tape
    counts this is built to match. ``base_10k`` has one such tape.
    """
    kept = manifest[manifest["keep"] & (manifest["place"] != "")]
    if kept.empty:
        return pd.Series("", index=manifest.index, dtype=str)

    # Each tape counted once, under the species that owns most of its clips. A tape
    # carrying two of them would otherwise appear in both pools, be dealt twice, and
    # have the second assignment overwrite the first, which breaks the tape counts this
    # exists to match. base_10k has one such tape.
    owner = kept.groupby("tape_id")["species"].agg(lambda values: values.value_counts().idxmax())
    counted = kept.drop_duplicates("tape_id").assign(owner=lambda f: f["tape_id"].map(owner))

    # Largest place first, and stably, so two places of equal size are always dealt in
    # the same order. The default sort is introsort and is not stable, so the comment
    # that used to sit here claiming a stable order was wrong.
    shape = counted.groupby(["place", "owner"])["tape_id"].nunique().unstack(fill_value=0)
    shape = shape.loc[shape.sum(axis=1).sort_values(ascending=False, kind="stable").index]

    generator = np.random.default_rng(seed)
    pools, taken = {}, dict.fromkeys(shape.columns, 0)
    for species in shape.columns:
        tapes = list(counted.loc[counted["owner"] == species, "tape_id"].unique())
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
            raise GroupingError(
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
    names = {site: min(group) for group in members.values() for site in group}

    return manifest["site"].map(lambda site: names.get(site, ""))
