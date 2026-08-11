"""Finding the model and the windows behind a result directory.

A result is named after the question rather than after the model, so explaining one
means working backwards from the directory name. Get that wrong quietly and the
tools read a checkpoint against windows the model never saw, which produces an
occlusion table that looks ordinary and means nothing.
"""

from __future__ import annotations

import pytest

from src.evaluate.explain import DEFAULT_CALL_TYPE_MODEL, ExplainError, _task_of, spec_for_result


class FakeDataset:
    species = ("SpermWhale", "KillerWhale", "HumpbackWhale")


class FakeConfig:
    dataset = FakeDataset()


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("cnn_small", "cnn_small"),
        ("cnn", "cnn"),
        ("xgboost", "xgboost"),
        ("calltype_spermwhale_coda_cnn_small", "cnn_small"),
        ("calltype_killerwhale_whistle_cnn", "cnn"),
    ],
)
def test_the_model_is_whichever_registry_name_the_directory_ends_with(result, expected):
    assert spec_for_result(result).name == expected


def test_a_call_type_result_naming_no_model_was_fitted_by_the_default():
    """``run_task`` leaves the default untagged, so a bare name means trees."""
    assert spec_for_result("calltype_spermwhale_coda").name == DEFAULT_CALL_TYPE_MODEL


def test_the_default_cannot_be_explained_by_band_even_though_it_now_saves():
    """The trees used to save nothing, so nothing could load them and that was the guard.

    They save now, because the prediction command needs a fitted fold it can read
    without retraining. The guard against explaining them moved rather than went away:
    a flat descriptor vector has no frequency axis, so masking a band means nothing
    there and the error names the representation that was asked.
    """
    from src.config import load_config
    from src.evaluate.explain import ExplainError, _frequencies

    trees = spec_for_result("calltype_spermwhale_coda")
    assert trees.load is not None, "the shipped fold would be unreadable"
    assert spec_for_result("cnn_small").load is not None

    with pytest.raises(ExplainError, match="no frequency axis"):
        _frequencies(load_config("configs/base.yaml"), trees)


@pytest.mark.parametrize(
    ("result", "species", "call_type"),
    [
        ("calltype_spermwhale_coda", "SpermWhale", "coda"),
        ("calltype_spermwhale_coda_cnn_small", "SpermWhale", "coda"),
        ("calltype_killerwhale_whistle", "KillerWhale", "whistle"),
        ("calltype_humpbackwhale_pulsed_call", "HumpbackWhale", "pulsed_call"),
    ],
)
def test_the_species_and_call_type_come_back_off_the_name(result, species, call_type):
    """Directory names are lowercased, so the species is matched rather than guessed."""
    assert _task_of(FakeConfig(), result) == (species, call_type)


def test_a_call_type_naming_no_species_under_study_is_refused():
    with pytest.raises(ExplainError, match="names no species"):
        _task_of(FakeConfig(), "calltype_belugawhale_click")
