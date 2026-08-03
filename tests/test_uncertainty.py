"""What an interval is allowed to claim.

Two mistakes would quietly flatter every result in the project. Resampling clips
instead of recordings treats near duplicates as fresh evidence and narrows the
interval; differencing unpaired folds produces a number that looks like a margin
and is not one. Both are pinned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.uncertainty import (
    ComparisonError,
    Interval,
    bootstrap_metric,
    fold_scores,
    paired_difference,
    shared_folds,
)

CLASSES = ["absent", "present"]


def predictions(tapes: int = 20, clips_per_tape: int = 10, seed: int = 0) -> pd.DataFrame:
    """Clip level predictions where every clip of a tape agrees, as real cuts do."""
    rng = np.random.default_rng(seed)
    rows = []
    for tape in range(tapes):
        label = tape % 2
        correct = rng.random() < 0.75
        confidence = 0.9 if correct else 0.2
        for cut in range(clips_per_tape):
            score = confidence if label == 1 else 1.0 - confidence
            rows.append(
                {
                    "clip_id": f"{tape:03d}{cut:03d}",
                    "tape_id": f"t{tape:03d}",
                    "label": label,
                    "p0": 1.0 - score,
                    "p1": score,
                }
            )
    return pd.DataFrame(rows)


def test_an_interval_reports_its_own_width_and_whether_it_spans_zero():
    assert Interval(0.2, 0.1, 0.3).excludes_zero
    assert not Interval(0.05, -0.1, 0.2).excludes_zero
    assert Interval(0.2, 0.1, 0.3).width == pytest.approx(0.2)


def test_resampling_draws_whole_recordings():
    """A tape is either wholly in a resample or wholly out; never split."""
    frame = predictions(tapes=12, clips_per_tape=8)
    interval = bootstrap_metric(frame, CLASSES, resamples=200, seed=1)

    assert interval.low <= interval.estimate <= interval.high
    assert interval.low >= 0.0 and interval.high <= 1.0


def test_resampling_clips_would_understate_the_interval():
    """The reason the bootstrap groups by tape, made explicit.

    Every clip of a tape carries the same prediction here, so treating clips as
    independent multiplies the apparent evidence tenfold and narrows the interval.
    """
    frame = predictions(tapes=16, clips_per_tape=10)
    by_tape = bootstrap_metric(frame, CLASSES, resamples=400, seed=2)

    per_clip = frame.assign(tape_id=frame["clip_id"])
    by_clip = bootstrap_metric(per_clip, CLASSES, resamples=400, seed=2)

    assert by_clip.width < by_tape.width, "resampling clips must look falsely precise"


def test_resampling_needs_something_to_resample():
    frame = predictions(tapes=1, clips_per_tape=5)
    with pytest.raises(ComparisonError, match="group"):
        bootstrap_metric(frame, CLASSES, resamples=10)


def test_resampling_refuses_a_frame_with_no_group_column():
    frame = predictions().drop(columns=["tape_id"])
    with pytest.raises(ComparisonError, match="tape_id"):
        bootstrap_metric(frame, CLASSES, resamples=10)


def test_resampling_refuses_missing_probability_columns():
    frame = predictions().drop(columns=["p1"])
    with pytest.raises(ComparisonError, match="probability"):
        bootstrap_metric(frame, CLASSES, resamples=10)


def test_the_same_seed_gives_the_same_interval():
    frame = predictions()
    first = bootstrap_metric(frame, CLASSES, resamples=200, seed=7)
    second = bootstrap_metric(frame, CLASSES, resamples=200, seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_a_paired_comparison_refuses_folds_that_do_not_match():
    left = pd.Series([0.8, 0.7, 0.9], index=[0, 1, 2])
    right = pd.Series([0.6, 0.5, 0.7], index=[0, 1, 3])
    with pytest.raises(ComparisonError, match="same folds"):
        paired_difference(left, right)


def test_a_clear_difference_is_resolved_and_a_noisy_one_is_not():
    folds = [0, 1, 2, 3, 4]
    control = pd.Series([0.50, 0.52, 0.48, 0.51, 0.49], index=folds)

    consistent = paired_difference(pd.Series([0.70, 0.72, 0.68, 0.71, 0.69], index=folds), control)
    assert consistent.resolved
    assert consistent.folds_agreeing == 5
    assert consistent.difference == pytest.approx(0.20, abs=1e-9)

    noisy = paired_difference(pd.Series([0.90, 0.20, 0.85, 0.25, 0.80], index=folds), control)
    assert not noisy.resolved
    assert noisy.low < 0 < noisy.high


def test_direction_agreement_is_counted_separately_from_the_p_value():
    """Four folds of five agreeing is worth reporting even when p is not small."""
    folds = [0, 1, 2, 3, 4]
    audio = pd.Series([0.60, 0.62, 0.61, 0.30, 0.63], index=folds)
    control = pd.Series([0.50, 0.50, 0.50, 0.50, 0.50], index=folds)

    result = paired_difference(audio, control)
    assert result.folds_agreeing == 4
    assert result.n_folds == 5


def repeated(differences: list[float], folds: int = 5) -> tuple[pd.Series, pd.Series]:
    """A model and its control whose per split differences are exactly as given."""
    index = pd.MultiIndex.from_tuples(
        [(i // folds, i % folds) for i in range(len(differences))], names=["repeat", "fold"]
    )
    control = pd.Series([0.5] * len(differences), index=index)
    return control + pd.Series(differences, index=index), control


def test_repeating_a_split_cannot_manufacture_significance():
    """The failure this project would otherwise have shipped.

    Ten repeats of a five fold split give fifty differences, and every one of them is
    measured on the same recordings. A plain paired t test treats those fifty as fifty
    independent observations, which drives the p value to nothing while the underlying
    spread has not changed at all. Repeats do buy something, because averaging over
    more splits does pin the mean down better, but nothing like a factor of ten.
    """
    once = [0.10, -0.02, 0.14, 0.05, -0.06]
    values = np.array(once * 10)
    fifty = paired_difference(*repeated(once * 10))

    naive = 2.0 * stats.t.sf(abs(values.mean() / np.sqrt(values.var(ddof=1) / 50)), 49)

    assert fifty.n_folds == 50
    assert naive < 0.01, "the naive test on these differences would call this decisive"
    assert not fifty.resolved, "ten copies of one split are not ten times the evidence"
    assert fifty.p_value > 20 * naive


def test_the_correction_uses_the_fold_overlap_and_not_the_row_count():
    """Nadeau and Bengio, spelled out on numbers the test can check by hand."""
    differences = [0.10, -0.02, 0.14, 0.05, -0.06]
    left, right = repeated(differences)
    result = paired_difference(left, right)

    values = np.array(differences)
    n, folds = len(values), 5
    corrected = (1.0 / n + 1.0 / (folds - 1)) * values.var(ddof=1)
    expected = 2.0 * stats.t.sf(abs(values.mean() / np.sqrt(corrected)), n - 1)

    assert result.p_value == pytest.approx(expected)

    naive = 2.0 * stats.t.sf(abs(values.mean() / np.sqrt(values.var(ddof=1) / n)), n - 1)
    assert result.p_value > naive, "the corrected test must be the more conservative one"


def test_the_interval_widens_with_the_correction_rather_than_the_p_value_alone():
    left, right = repeated([0.10, -0.02, 0.14, 0.05, -0.06])
    corrected = paired_difference(left, right)
    naive = stats.ttest_rel(left.to_numpy(), right.to_numpy()).confidence_interval(0.95)

    assert corrected.low < naive.low and corrected.high > naive.high


def test_a_comparison_needs_more_than_one_fold():
    single = pd.Series([0.8], index=[0])
    with pytest.raises(ComparisonError, match="fold"):
        paired_difference(single, pd.Series([0.6], index=[0]))


def test_identical_scores_do_not_blow_up():
    folds = [0, 1, 2]
    same = pd.Series([0.5, 0.5, 0.5], index=folds)
    result = paired_difference(same, same)
    assert result.difference == 0.0
    assert not result.resolved


def test_fold_scores_pair_on_repeat_and_fold_together():
    metrics = pd.DataFrame(
        {
            "repeat": [0, 0, 1, 1],
            "fold": [0, 1, 0, 1],
            "macro_f1": [0.5, 0.6, 0.7, 0.8],
        }
    )
    scores = fold_scores(metrics)

    assert list(scores.index.names) == ["repeat", "fold"]
    assert scores.loc[(1, 0)] == 0.7


def test_a_run_from_before_repeats_is_read_as_repeat_zero():
    metrics = pd.DataFrame({"fold": [0, 1], "macro_f1": [0.5, 0.6]})

    assert list(fold_scores(metrics).index) == [(0, 0), (0, 1)]


def test_an_old_run_pairs_with_the_first_repeat_of_a_new_one():
    """The network was fitted on one split while the trees ran ten.

    Repeat zero of the repeated run is that same split, so the two are genuinely
    paired on it. Reading the old run as repeat zero is what lets them meet.
    """
    old = fold_scores(pd.DataFrame({"fold": [0, 1, 2], "macro_f1": [0.5, 0.6, 0.7]}))
    repeated = fold_scores(
        pd.DataFrame(
            {
                "repeat": [0, 0, 0, 1, 1, 1],
                "fold": [0, 1, 2, 0, 1, 2],
                "macro_f1": [0.4, 0.5, 0.6, 0.9, 0.9, 0.9],
            }
        )
    )
    left, right = shared_folds(old, repeated)

    assert len(left) == 3, "only the split they share can be paired"
    assert list(right) == [0.4, 0.5, 0.6]


def test_two_runs_sharing_no_split_are_refused():
    left = fold_scores(pd.DataFrame({"repeat": [0, 0], "fold": [0, 1], "macro_f1": [0.5, 0.6]}))
    right = fold_scores(pd.DataFrame({"repeat": [7, 7], "fold": [0, 1], "macro_f1": [0.5, 0.6]}))

    with pytest.raises(ComparisonError, match="nothing can be paired"):
        shared_folds(left, right)


def test_fold_scores_refuse_a_metric_that_is_not_there():
    with pytest.raises(ComparisonError, match="macro_f1"):
        fold_scores(pd.DataFrame({"fold": [0], "accuracy": [0.5]}))
