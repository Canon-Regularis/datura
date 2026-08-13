"""What a run is scored on, and how many times over.

Repeats exist because five folds cannot separate the differences this project
reports. They are only worth anything if each repeat is a genuinely different
split, so that is what these tests hold: repeat zero must reproduce the single
split exactly, and every later repeat must move the recordings around.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.config import Config
from src.data.splits import folds_for_index
from src.train.folds import FoldPlan
from tests.helpers import clip_rows


def window_index(tapes_per_species: int = 8, clips_per_tape: int = 3) -> pd.DataFrame:
    return clip_rows(tapes_per_species, clips_per_tape)


@pytest.fixture
def cfg() -> Config:
    from src.config import load_config

    return load_config("configs/base.yaml")


def test_a_single_plan_yields_one_repeat_of_the_folds_it_was_given(cfg):
    folds = folds_for_index(window_index(), cfg)
    plan = FoldPlan.single(folds)

    assert [(repeat, built) for repeat, built in plan] == [(0, folds)]


def test_repeat_zero_reproduces_the_single_split(cfg):
    index = window_index()
    plan = FoldPlan.repeated(cfg, index, repeats=3)

    _, first = next(iter(plan))
    expected = folds_for_index(index, cfg)
    assert [fold.test_clips for fold in first] == [fold.test_clips for fold in expected]


def test_every_repeat_holds_out_different_recordings(cfg):
    plan = FoldPlan.repeated(cfg, window_index(), repeats=4)

    held_out = [tuple(sorted(fold.test_clips)) for _, folds in plan for fold in folds]
    assert len(set(held_out)) > len(held_out) // 2, (
        "repeats must move the test sets, not reuse them"
    )


def test_repeats_yield_that_many_splits(cfg):
    plan = FoldPlan.repeated(cfg, window_index(), repeats=5)

    assert [repeat for repeat, _ in plan] == [0, 1, 2, 3, 4]
    assert all(len(folds) == cfg.split.n_folds for _, folds in plan)


def test_a_plan_of_no_repeats_is_refused():
    with pytest.raises(ValueError, match="at least one repeat"):
        FoldPlan(build=lambda _: [], repeats=0)


def test_a_repeat_moves_the_split_seed_and_nothing_else(cfg):
    """Which is why a repeat that returns the same partition adds nothing at all.

    The warning used to say such repeats vary the model, which reads as a second draw
    worth having. They do not: a model takes its seed from its own config and never
    sees the repeat index, so a partition that comes back the same brings the same
    everything with it. Refitting the probe under the shuffled place folds showed it,
    with fold zero scoring 0.443 on repeat 0 and 0.443 again on repeat 3.

    A fixture cannot demonstrate the collapse. It needs the real group sizes, and every
    synthetic index tried here redraws cleanly, so what is pinned is the mechanism.
    """
    index = window_index()
    plan = FoldPlan.repeated(cfg, index, repeats=3)

    for repeat in range(3):
        shifted = replace(cfg, split=replace(cfg.split, seed=cfg.split.seed + repeat))
        expected = folds_for_index(index, shifted)
        built = plan.build(repeat)
        assert [fold.test_clips for fold in built] == [fold.test_clips for fold in expected]
        assert [fold.train_clips for fold in built] == [fold.train_clips for fold in expected]
