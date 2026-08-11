"""Holding out a recording context rather than a recording.

A tape grouped fold proves no recording sits on both sides of the boundary. It does
not prove the model survives a site, a hydrophone or a recording chain it has never
heard, and those are different questions. Cuts from two tapes made at the same place,
on the same trip, through the same equipment share far more than the animal.

``context_10k`` differs from ``base_10k`` in one line of configuration, so the gap
between the two reports is attributable to the fold rule and to nothing else. That
"nothing else" is what most of this file checks.

The group is the physical place, and both weaker versions of it are tested here because
both leaked. Grouping on the raw site put whole tapes on both sides of a fold, since six
tapes carry clips the notes place at more than one site. Merging the sites a tape links
fixed that and left a second leak: names no tape connects stayed separate, so six groups
were all Dominica and 52% of held out clips sat in a fold whose place was also in
training. Only the third version holds a place out.
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
    assert context.split.group_column == "place"


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


def test_a_place_is_never_split_across_a_fold():
    """The grouping the experiment actually uses, and the leak that forced it.

    A context merges only the sites some tape links, so `Dominica` and `DOMINICA` stayed
    apart and more than half the held out clips had their place in the training set. The
    place column merges them by name, and holding one out is what the section claims.
    """
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = with_a_group(load_manifest(cfg), "place")

    assert manifest.groupby("tape_id")["place"].nunique().max() == 1
    assert manifest["place"].nunique() < manifest["context"].nunique(), "the merge did nothing"

    place_of = manifest.set_index("clip_id")["place"]
    tape_of = manifest.set_index("clip_id")["tape_id"]
    for fold in make_folds(manifest, cfg):
        for column in (place_of, tape_of):
            held_out = set(column.loc[list(fold.test_clips)])
            trained_on = set(column.loc[list(fold.fitting_clips)])
            assert not held_out & trained_on, f"fold {fold.index} is scored on what it saw"


def test_the_context_grouping_it_replaced_did_leak_a_place():
    """Recorded because the number it produced was published before this was measured.

    A place split that is half leaked reports a fifth of the effect, and nothing about
    the folds looks wrong while it does.
    """
    from dataclasses import replace

    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    # Folds built on the grouping this replaced, rather than on the one in the config.
    on_context = replace(cfg, split=replace(cfg.split, group_column="context"))
    manifest = with_a_group(load_manifest(cfg), "context")
    place_of = manifest.set_index("clip_id")["place"]

    leaked = held = 0
    for fold in make_folds(manifest, on_context):
        trained_on = set(place_of.loc[list(fold.fitting_clips)])
        places = place_of.loc[list(fold.test_clips)]
        leaked += int(places.isin(trained_on).sum())
        held += len(places)
    assert leaked / held > 0.4, "the context grouping leaked half its test set and this says so"


def test_a_context_never_splits_a_tape():
    """The check that changed the fold rule, and the reason it is not the raw site.

    Grouping on the site alone leaked one to three whole tapes across every fold,
    because six tapes are annotated with more than one site. Cuts from one tape are
    near duplicates, so those folds measured memorisation of a recording and reported
    it as generalisation to a new place.
    """
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = with_a_group(load_manifest(cfg), "context")

    per_tape = manifest.groupby("tape_id")["context"].nunique()
    assert per_tape.max() == 1, "a tape spanning two contexts can be split across a fold"

    # And the leak this replaced is real rather than hypothetical, so the stronger
    # rule is not paying for a problem nobody had.
    by_site = with_a_group(load_manifest(cfg), "site")
    assert by_site.groupby("tape_id")["site"].nunique().max() > 1

    tape_of = manifest.set_index("clip_id")["tape_id"]
    for fold in make_folds(manifest, cfg):
        held_out = set(tape_of.loc[list(fold.test_clips)])
        trained_on = set(tape_of.loc[list(fold.fitting_clips)])
        assert not held_out & trained_on, f"fold {fold.index} trains on a tape it is scored on"


def test_a_context_merges_only_sites_a_tape_links():
    """Otherwise it would be quietly coarsening the experiment it is meant to sharpen."""
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = with_a_group(load_manifest(cfg), "context")

    merged = manifest.groupby("context")["site"].nunique()
    assert merged.max() > 1, "no context merges anything, so the union did not run"
    assert manifest["context"].nunique() == 39
    assert manifest["site"].nunique() == 47

    for context, group in manifest[manifest["context"].isin(merged[merged > 1].index)].groupby(
        "context"
    ):
        shared = group.groupby("tape_id")["site"].nunique()
        assert shared.max() > 1, f"{context} merged sites that no tape connects"


def test_a_clip_with_no_site_cannot_join_a_group():
    """A blank is not a group.

    Every clip whose site the notes never recorded would otherwise be collected into
    one pseudo group and held out together, which is neither a recording context nor
    a random sample of one.
    """
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = load_manifest(cfg)

    grouped = with_a_group(manifest, "place")
    assert len(grouped) < len(manifest), "some base_10k clips carry no site"
    assert (grouped["place"].str.strip() != "").all()

    # The tape is parsed from the filename and is never blank, so base passes through.
    assert len(with_a_group(manifest, "tape_id")) == len(manifest)


def test_no_context_appears_on_both_sides_of_a_fold():
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = load_manifest(cfg)

    folds = make_folds(manifest, cfg)
    assert_no_group_leak(with_a_group(manifest, "context"), folds, "context")


def test_every_fold_holds_every_species():
    """The assumption the whole experiment rests on.

    Humpback spans exactly five of the twenty four places, which is the fewest a five
    fold split can use, so this is the finest grouping the corpus supports. If one fold
    ever lost a species, it would be scoring a class that was never tested and macro-F1
    would say so silently.
    """
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = load_manifest(cfg)
    by_clip = with_a_group(manifest, "place").set_index("clip_id")

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


def test_folds_can_be_built_from_a_window_index_that_has_no_context():
    """The entry point training actually uses, which is not the one that reads a manifest.

    ``folds_for_index`` receives the feature cache index, and extraction writes only
    the clip identity and the audio header there. A group taken from the field notes
    has to be joined on. Testing ``make_folds`` against a manifest passes while this
    path raises, which is exactly what happened.
    """
    from src.data.splits import folds_for_index

    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = load_manifest(cfg)

    # A window index carries what src/features/extract.py writes, and no more.
    index = manifest[["clip_id", "tape_id", "species", "label"]].copy()
    index["window_index"] = 0
    assert "context" not in index.columns

    folds = folds_for_index(index, cfg)
    assert len(folds) == cfg.split.n_folds
    assert_no_group_leak(with_a_group(manifest, "context"), folds, "context")


def test_every_grouping_column_regenerates_from_committed_code():
    """A configuration cannot group on a column only an ad hoc script can produce.

    ``place_shuffled`` was dealt by hand at first, which meant a fresh clone running
    the manifest stage would build a corpus the control config could not be fitted on,
    and nothing would have said so until a fold failed.
    """
    from src.data.manifest import recording_contexts, recording_places, shuffled_places

    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    manifest = load_manifest(cfg, kept_only=False)

    rebuilt = {
        "context": recording_contexts(manifest),
        "place": recording_places(manifest),
        "place_shuffled": shuffled_places(manifest, cfg.split.seed),
    }
    for column, values in rebuilt.items():
        assert column in manifest.columns, f"{column} is not in the committed manifest"
        assert values.equals(manifest[column]), f"{column} does not regenerate to what is committed"

    assert cfg.split.group_column in rebuilt, "the fold column is one of these"


def test_the_control_matches_the_real_split_it_controls_for():
    """A control that changed the group sizes would measure something else entirely."""
    cfg = load_config(CONTEXT)
    corpus_present(cfg)
    kept = load_manifest(cfg, kept_only=True)
    real = kept[kept["place"] != ""]
    pseudo = kept[kept["place_shuffled"] != ""]

    assert real["place"].nunique() == pseudo["place_shuffled"].nunique()
    assert sorted(real.groupby("place")["tape_id"].nunique()) == sorted(
        pseudo.groupby("place_shuffled")["tape_id"].nunique()
    ), "the pseudo places do not hold the same number of tapes"
    assert sorted(real.groupby("place")["species"].nunique()) == sorted(
        pseudo.groupby("place_shuffled")["species"].nunique()
    ), "the pseudo places do not carry the same species mix"
    assert pseudo.groupby("tape_id")["place_shuffled"].nunique().max() == 1
