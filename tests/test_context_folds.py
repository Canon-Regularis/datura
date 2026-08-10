"""Holding out a recording context rather than a recording.

A tape grouped fold proves no recording sits on both sides of the boundary. It does
not prove the model survives a site, a hydrophone or a recording chain it has never
heard, and those are different questions. Cuts from two tapes made at the same place,
on the same trip, through the same equipment share far more than the animal.

``context_10k`` differs from ``base_10k`` in one line of configuration, so the gap
between the two reports is attributable to the fold rule and to nothing else. That
"nothing else" is what most of this file checks.
"""

from __future__ import annotations

import pytest
import yaml

from src.config import PROJECT_ROOT, load_config
from src.data.manifest import load_manifest, manifest_path
from src.data.splits import assert_no_group_leak, make_folds, with_a_group

CONTEXT = "configs/context.yaml"
BASE = "configs/base.yaml"


def corpus_present(cfg) -> None:
    if not manifest_path(cfg).exists():
        pytest.skip(f"{manifest_path(cfg).name} absent; run python -m src.data.manifest first")


def test_the_two_configs_differ_only_in_what_a_fold_boundary_means():
    """Otherwise the gap between them would measure more than one thing."""
    base, context = load_config(BASE), load_config(CONTEXT)

    assert base.dataset.species == context.dataset.species
    assert base.audio == context.audio
    assert base.spectrogram == context.spectrogram
    assert base.encoder == context.encoder
    assert base.split.n_folds == context.split.n_folds
    assert base.split.seed == context.split.seed

    assert base.split.group_column == "tape_id"
    assert context.split.group_column == "site"


def test_the_context_experiment_reads_the_base_corpus():
    """Same data, so it must not build a second manifest or a second feature cache."""
    context = load_config(CONTEXT)

    assert context.name == "context_10k", "results are keyed by the experiment"
    assert context.corpus == "base_10k", "artifacts are keyed by the corpus"
    assert context.audio_digest == load_config(BASE).audio_digest
    assert manifest_path(context) == manifest_path(load_config(BASE))


def test_no_duplicate_corpus_artifacts_were_left_behind():
    """A configuration that borrows a corpus must not also ship one."""
    stray = sorted(p.name for p in (PROJECT_ROOT / "data" / "metadata").glob("*context_10k*"))
    assert not stray, f"{stray} duplicates the base_10k corpus"


def test_a_clip_with_no_site_cannot_join_a_group():
    """A blank is not a group.

    Every clip whose site the notes never recorded would otherwise be collected into
    one pseudo group and held out together, which is neither a recording context nor
    a random sample of one.
    """
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = load_manifest(cfg)

    grouped = with_a_group(manifest, "site")
    assert len(grouped) < len(manifest), "some base_10k clips carry no site"
    assert (grouped["site"].str.strip() != "").all()

    # The tape is parsed from the filename and is never blank, so base passes through.
    assert len(with_a_group(manifest, "tape_id")) == len(manifest)


def test_no_site_appears_on_both_sides_of_a_fold():
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = load_manifest(cfg)

    folds = make_folds(manifest, cfg)
    assert_no_group_leak(with_a_group(manifest, "site"), folds, "site")


def test_every_fold_holds_every_species():
    """The assumption the whole experiment rests on.

    Each species spans at least eleven of the forty seven sites, so a five fold site
    grouped split can put all three in every fold. If one ever could not, a fold would
    be scoring a class that was never tested and macro-F1 would say so silently.
    """
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = load_manifest(cfg)
    by_clip = with_a_group(manifest, "site").set_index("clip_id")

    for fold in make_folds(manifest, cfg):
        for part, clips in (
            ("train", fold.train_clips),
            ("test", fold.test_clips),
        ):
            present = set(by_clip.loc[list(clips), "species"])
            assert present == set(cfg.dataset.species), (
                f"fold {fold.index} {part} is missing {set(cfg.dataset.species) - present}"
            )


def test_grouping_by_collection_code_is_refused_rather_than_attempted():
    """Recorded as a finding, because it looks reasonable and deletes a class.

    There are seven codes across these three species and HumpbackWhale has exactly
    one. Holding out a code removes the class from a fold entirely, so the split that
    looks like a stronger version of this experiment is not available.
    """
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = load_manifest(cfg)
    coded = with_a_group(manifest, "collection_code")

    per_species = coded.groupby("species")["collection_code"].nunique()
    assert per_species.min() == 1, "a species with one code cannot survive a code grouped fold"
    assert per_species.idxmin() == "HumpbackWhale"


def test_the_context_config_trains_the_models_that_can_measure_a_gap():
    """Only models carrying fifty splits in both reports can compare against base."""
    raw = yaml.safe_load((PROJECT_ROOT / CONTEXT).read_text(encoding="utf-8"))
    declared = set(raw["pipeline"]["models"])

    assert {"xgboost", "probe", "logbook", "metadata"} <= declared
    assert not {"cnn", "cnn_small"} & declared, "the five fold networks cannot measure this gap"
