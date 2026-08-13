"""Which results are comparable with which.

A score in this project means nothing without the control it was measured against,
so a result that belongs to no family has to stop the report rather than be printed
beside numbers it cannot be compared with.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest

from src.evaluate.families import SPECIES_FAMILY, FamilyError, discover, result_names
from src.results import model_directory
from tests.helpers import SPECIES

BINARY = ("absent", "present")


def write_result(config, name: str, class_names: Sequence[str]) -> None:
    """The two files a family needs to see: a summary and a confusion matrix."""
    directory = model_directory(config, name)
    directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"metric": ["macro_f1"], "mean": [0.5], "std": [0.1]}).to_csv(
        directory / "summary.csv", index=False
    )
    size = len(class_names)
    pd.DataFrame([[1] * size for _ in class_names], index=class_names, columns=class_names).to_csv(
        directory / "confusion.csv"
    )


def test_no_results_means_no_families(config):
    assert result_names(config) == []
    assert discover(config) == []


def test_the_species_models_are_one_family_against_the_metadata_control(config):
    for name in ("xgboost", "cnn", "metadata"):
        write_result(config, name, SPECIES)

    families = discover(config)

    assert [family.key for family in families] == [SPECIES_FAMILY]
    assert set(families[0].members) == {"xgboost", "cnn"}
    assert families[0].control == "metadata"


def test_a_call_type_is_its_own_family_against_its_context_control(config):
    write_result(config, "calltype_spermwhale_coda", BINARY)
    write_result(config, "calltype_spermwhale_coda_context", BINARY)

    families = discover(config)

    assert len(families) == 1
    assert families[0].title == "SpermWhale, coda"
    assert families[0].control == "calltype_spermwhale_coda_context"
    assert families[0].class_names == ("absent", "present")


def test_a_second_model_on_the_same_task_joins_that_family(config):
    write_result(config, "calltype_spermwhale_coda", BINARY)
    write_result(config, "calltype_spermwhale_coda_cnn_small", BINARY)
    write_result(config, "calltype_spermwhale_coda_context", BINARY)

    families = discover(config)

    assert len(families) == 1
    assert set(families[0].members) == {
        "calltype_spermwhale_coda",
        "calltype_spermwhale_coda_cnn_small",
    }


def test_one_task_cannot_steal_a_sibling_whose_name_extends_it(config):
    """The quiet failure: a margin taken against the wrong control, saying nothing.

    A model is named after its task with the model appended, so "call" is a prefix of
    "call response". Matching on the prefix would hand every call response model to the
    call family and compare it against the wrong floor.
    """
    for task in ("calltype_killerwhale_call", "calltype_killerwhale_call_response"):
        write_result(config, task, BINARY)
        write_result(config, f"{task}_cnn_small", BINARY)
        write_result(config, f"{task}_context", BINARY)

    by_key = {family.key: family for family in discover(config)}

    assert set(by_key["calltype_killerwhale_call"].members) == {
        "calltype_killerwhale_call",
        "calltype_killerwhale_call_cnn_small",
    }
    assert set(by_key["calltype_killerwhale_call_response"].members) == {
        "calltype_killerwhale_call_response",
        "calltype_killerwhale_call_response_cnn_small",
    }


def test_a_result_with_no_control_stops_the_report(config):
    write_result(config, "calltype_killerwhale_squeal", BINARY)

    with pytest.raises(FamilyError, match="no control"):
        discover(config)


def test_species_models_without_their_control_stop_the_report(config):
    write_result(config, "xgboost", SPECIES)

    with pytest.raises(FamilyError, match="metadata"):
        discover(config)


def test_a_control_with_nothing_on_trial_stops_the_report(config):
    write_result(config, "calltype_spermwhale_coda_context", BINARY)

    with pytest.raises(FamilyError, match="no results"):
        discover(config)


def test_two_controls_for_one_task_stop_the_report(config):
    """A leftover run leaves two floors, and the margin would depend on the guess."""
    write_result(config, "calltype_spermwhale_coda", BINARY)
    write_result(config, "calltype_spermwhale_coda_context", BINARY)
    write_result(config, "calltype_spermwhale_coda_context_short", BINARY)

    with pytest.raises(FamilyError, match="two controls"):
        discover(config)


def test_a_result_claimed_by_two_families_stops_the_report(config):
    """A model measured against two different floors has two different margins.

    ``calltype_spermwhale_context`` is a task with no call type, which nothing can
    produce: ``Task`` always has one. It stands in for a leftover directory, and what
    matters is that the report refuses rather than picking a control. Which refusal it
    is depends on how far the name gets through the grammar, so both are accepted.
    """
    write_result(config, "calltype_spermwhale_coda", BINARY)
    write_result(config, "calltype_spermwhale_coda_context", BINARY)
    write_result(config, "calltype_spermwhale_context", BINARY)

    with pytest.raises(FamilyError, match=r"two different controls|no results|no control"):
        discover(config)


def test_classes_are_read_from_the_result_rather_than_assumed(config):
    write_result(config, "xgboost", SPECIES)
    write_result(config, "metadata", SPECIES)
    write_result(config, "calltype_spermwhale_coda", BINARY)
    write_result(config, "calltype_spermwhale_coda_context", BINARY)

    by_key = {family.key: family for family in discover(config)}

    assert by_key[SPECIES_FAMILY].class_names == SPECIES
    assert by_key["calltype_spermwhale_coda"].class_names == ("absent", "present")


def test_the_result_name_grammar_round_trips():
    """One grammar, one parser. It used to have four.

    ``train.tasks`` and ``train.calltypes`` built these names, ``families`` took them
    apart three times and ``explain`` twice, each with its own loop over the species
    list and its own rule for stripping a model suffix. Any disagreement between them
    attaches a result to the wrong control, which is a margin measured against the
    wrong thing with nothing on the page to say so.
    """
    from src.results import ResultName

    species = ["HumpbackWhale", "SpermWhale", "KillerWhale"]
    known = ["xgboost", "cnn", "cnn_small", "metadata", "logbook", "probe"]

    for name in (
        "xgboost",
        "xgboost+probe",
        "calltype_spermwhale_coda",
        "calltype_spermwhale_coda_context",
        "calltype_spermwhale_coda_cnn_small",
        "calltype_killerwhale_pulsed_call_context",
    ):
        assert ResultName.parse(name, species=species, models=known).render() == name


def test_the_longest_model_name_wins_when_one_ends_with_another():
    """``cnn`` is a suffix of ``cnn_small``, and the shorter one must not claim it."""
    from src.results import ResultName

    parsed = ResultName.parse(
        "calltype_spermwhale_coda_cnn_small",
        species=["SpermWhale"],
        models=["cnn", "cnn_small"],
    )
    assert parsed.model == "cnn_small"
    assert parsed.call_type == "coda"


def test_a_name_for_no_species_under_study_is_refused():
    from src.results import ResultName

    with pytest.raises(ValueError, match="names no species"):
        ResultName.parse("calltype_belugawhale_click", species=["SpermWhale"], models=[])
