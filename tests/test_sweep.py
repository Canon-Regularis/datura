"""Choosing a representation without choosing it on the published number.

A spectrogram setting is a hyperparameter. Picking one by looking at the test folds is
the same mistake as picking a decision threshold there, and it is harder to notice
because it looks like engineering rather than like cheating.

These tests hold the three properties that make the choice honest: the ranking reads the
validation scores, a candidate cannot reach the published tables while it is still a
candidate, and a leader whose margin is inside what these folds resolve never causes the
test folds to be read at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import experiment_configs, load_config
from src.evaluate import sweep
from tests.helpers import needs


def test_the_sweep_settings_are_invisible_to_the_published_set():
    """A candidate must not reach the multiplicity correction or the reproduce job.

    ``experiment_configs`` globs ``configs/*.yaml`` and is not recursive, which is the
    only reason ``configs/sweep/`` is a directory rather than a naming convention. A
    setting discovered there would publish a representation that exists to be discarded.
    """
    discovered = {path.name for path in experiment_configs()}
    settings = {path.name for path in sweep.SWEEP_DIRECTORY.glob("*.yaml")}

    assert settings, "no sweep settings to check"
    assert not (settings & discovered), f"{sorted(settings & discovered)} would be published"


def test_every_setting_moves_one_axis_from_the_baseline():
    """One axis at a time, or a win cannot be attributed to anything.

    Each setting extends base.yaml, so anything it does not name it inherits, and the
    difference between it and the baseline is the thing being tested.
    """
    base = load_config(sweep.BASELINE)
    axes = {
        "n_fft": lambda c: c.spectrogram.n_fft,
        "n_mels": lambda c: c.spectrogram.n_mels,
        "window_seconds": lambda c: c.audio.window_seconds,
        "compression": lambda c: c.spectrogram.compression,
    }

    for path in sorted(sweep.SWEEP_DIRECTORY.glob("*.yaml")):
        cfg = load_config(path)
        moved = [name for name, read in axes.items() if read(cfg) != read(base)]
        assert cfg.corpus == base.corpus, f"{path.name} reads a different corpus"
        assert cfg.dataset.species == base.dataset.species, f"{path.name} changes the classes"
        # The window carries its hop, which is not a separate axis.
        assert len(moved) == 1, f"{path.name} moves {moved} rather than one axis"


def test_the_ranking_reads_validation_and_not_test(monkeypatch):
    """The property the phase rests on.

    Fed a candidate that is worse on validation and better on test, the ranking has to
    prefer the other one. Ranking on test would make this pass while measuring nothing.
    """

    def fake_mean(path, metric):
        if "validation" in path.name:
            return (0.9, 0.0, 2) if "sweep_nfft2048" in str(path) else (0.5, 0.0, 2)
        return (0.1, 0.0, 2) if "sweep_nfft2048" in str(path) else (0.99, 0.0, 2)

    monkeypatch.setattr(sweep, "_mean", fake_mean)
    table = sweep.compare()

    assert table.iloc[0]["setting"] == "sweep_nfft2048", (
        "the setting that won on validation has to win, however it did on test"
    )


def test_it_says_what_to_run_when_nothing_is_fitted(monkeypatch):
    """A sweep with no results is the normal state of a fresh clone, not an error to debug."""
    monkeypatch.setattr(sweep, "_mean", lambda path, metric: None)

    with pytest.raises(sweep.NothingToCompare, match="fit one with"):
        sweep.compare()


def test_the_baseline_is_one_of_the_candidates():
    """Otherwise the sweep can only report which change is least bad."""
    assert sweep.BASELINE in sweep.candidates()
    assert sweep.BASELINE.exists()


def test_an_unresolved_leader_never_reaches_the_test_folds(monkeypatch):
    """The guard that keeps the sweep from spending its one look at test on noise.

    Every table has a leader, and with folds that overlap as much as these do a margin
    of a couple of points is a coin flip. Reading test to see how the coin landed is
    how a sweep turns into a search over the published number, so the read has to be
    unreachable rather than merely discouraged.
    """
    table = pd.DataFrame([{"setting": "sweep_nfft2048", "validation_macro_f1": 0.9}])
    monkeypatch.setattr(sweep, "compare", lambda model, metric: table)
    monkeypatch.setattr(sweep, "margin_over_baseline", lambda *a: (0.01, -0.04, 0.06, 0.62, 50))

    def forbidden(*args, **kwargs):
        raise AssertionError("test folds were read for a leader that is not separable")

    monkeypatch.setattr(sweep, "clip_metrics_path", forbidden)
    sweep.report()


def test_a_separable_leader_does_reach_the_test_folds(monkeypatch):
    """The other half, or the guard above would pass on a function that never reads test."""
    table = pd.DataFrame([{"setting": "sweep_nfft2048", "validation_macro_f1": 0.9}])
    monkeypatch.setattr(sweep, "compare", lambda model, metric: table)
    monkeypatch.setattr(sweep, "margin_over_baseline", lambda *a: (0.09, 0.04, 0.14, 0.004, 50))
    seen = []
    monkeypatch.setattr(sweep, "clip_metrics_path", lambda cfg, model: seen.append(cfg.name))
    monkeypatch.setattr(sweep, "_mean", lambda path, metric: None)

    sweep.report()

    assert seen == ["sweep_nfft2048"], "a separable leader has to be measured on test"


@pytest.mark.slow
def test_the_winner_is_reported_against_the_baseline():
    """Runs only once something has been fitted, and skips honestly before that."""
    from src.results import validation_metrics_path

    needs(
        validation_metrics_path(load_config(sweep.BASELINE), "xgboost_centred"),
        "refit the baseline so it records its validation folds",
    )
    table = sweep.compare()
    assert "validation_macro_f1" in table.columns
    assert table["validation_macro_f1"].is_monotonic_decreasing
