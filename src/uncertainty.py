"""How much of a reported difference the design can actually resolve.

Every score in this project rests on a handful of independent recordings. Reporting a
mean and a spread across folds understates how little that settles. This module
supplies the two things a margin needs before it can be read as a result: an interval
that respects the unit of independence, and a paired test against the control the
model was measured with.

Three rules matter here more than the arithmetic.

Resampling draws whole recordings. Clips cut from one tape are near duplicates, so
resampling clips would treat each near duplicate as fresh evidence and produce an
interval several times too narrow. The same reasoning that makes folds group by
tape makes the bootstrap group by tape.

The comparison is paired. A model and its control are fitted on identical folds, so
the fold is the natural pairing, and using it removes the variance that comes from
some folds simply being harder than others.

The paired test is corrected for the overlap between folds. Two folds of a five fold
split share three quarters of their training data, and repeating the split reuses
every clip again, so the fold differences are correlated. A plain paired t test treats
them as independent and reports a standard error roughly three times too small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src import scoring
from src.errors import DaturaError

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
GROUP_COLUMN = "tape_id"


class ComparisonError(DaturaError):
    """Raised when two sets of scores cannot honestly be compared."""


@dataclass(frozen=True)
class Interval:
    """A point estimate with the range the data supports."""

    estimate: float
    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}]"

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


@dataclass(frozen=True)
class PairedComparison:
    """One model against its control, fold by fold.

    ``folds_agreeing`` is reported alongside the p value because it says something
    the p value does not. Four folds of five pointing the same way is worth knowing
    even when five paired observations cannot reach significance.
    """

    difference: float
    low: float
    high: float
    p_value: float
    folds_agreeing: int
    n_folds: int

    @property
    def resolved(self) -> bool:
        """Whether this design separates the two at the usual threshold.

        A comparison with no spread across folds carries no p value, and a comparison
        against nothing is not resolved.
        """
        return bool(self.p_value < 0.05)

    def describe(self) -> str:
        direction = "higher" if self.difference >= 0 else "lower"
        if np.isnan(self.p_value):
            verdict = f"every one of {self.n_folds} folds differed by the same amount, so "
            verdict += "there is no spread to test"
            return f"{abs(self.difference):.3f} {direction}, {verdict}"
        verdict = (
            "resolved at p < 0.05"
            if self.resolved
            else f"not resolvable by this design, on {self.n_folds} folds"
        )
        return (
            f"{abs(self.difference):.3f} {direction} "
            f"[{self.low:+.3f}, {self.high:+.3f}], p = {self.p_value:.3f}, "
            f"{verdict}; the direction held in {self.folds_agreeing} of {self.n_folds} folds"
        )


def _folds_per_repeat(index: pd.Index) -> int:
    """How many folds one split was cut into.

    A repeated run is indexed by repeat and fold, and it is the fold count of a single
    split that sets how much training data two folds share. A run with a plain index
    is one split, so every row is a fold of it.
    """
    if isinstance(index, pd.MultiIndex) and "fold" in (index.names or []):
        return int(index.get_level_values("fold").nunique())
    return len(index)


def _variance_of_the_mean(differences: np.ndarray, folds_per_repeat: int) -> float:
    """How much the mean difference would move if the whole thing were run again.

    The obvious answer is the sample variance over the number of folds, and it is
    wrong here. Any two folds of a cross validation are fitted on training sets that
    overlap in most of their data, and repeats reuse every clip again, so the fold
    differences are correlated rather than independent. Dividing by the count treats
    fifty correlated estimates as fifty independent ones and shrinks the standard
    error by roughly a factor of three.

    Nadeau and Bengio's correction replaces ``1 / n`` with ``1 / n + n_test /
    n_train``, which for k folds is ``1 / n + 1 / (k - 1)``. The second term does not
    fall as repeats are added, which is the point: repeating a split measures the same
    recordings more precisely, and it does not go and find more whales.
    """
    if folds_per_repeat < 2:
        raise ComparisonError(f"cannot correct a comparison over {folds_per_repeat} fold")
    overlap = 1.0 / (folds_per_repeat - 1)
    return (1.0 / len(differences) + overlap) * float(differences.var(ddof=1))


def _percentile_bounds(confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    return 100.0 * tail, 100.0 * (1.0 - tail)


def bootstrap_metric(
    predictions: pd.DataFrame,
    class_names: list[str],
    *,
    metric: str = "macro_f1",
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
    group_column: str = GROUP_COLUMN,
) -> Interval:
    """Resample recordings with replacement and rescore, giving an interval.

    ``predictions`` is a clip level frame as written by the runner: one row per
    clip, carrying its group, its label and a probability per class.
    """
    if group_column not in predictions.columns:
        raise ComparisonError(f"predictions have no {group_column} column to resample over")

    columns = scoring.probability_columns(len(class_names))
    missing = set(columns) - set(predictions.columns)
    if missing:
        raise ComparisonError(f"predictions are missing probability columns: {sorted(missing)}")

    rows_by_group = {
        group: frame.index.to_numpy()
        for group, frame in predictions.groupby(group_column, sort=True)
    }
    groups = np.array(sorted(rows_by_group))
    if len(groups) < 2:
        raise ComparisonError(f"cannot resample over {len(groups)} {group_column} group")

    labels = predictions["label"].to_numpy()
    probabilities = predictions[columns].to_numpy()

    # Argmax once. The resamples only ever reorder these rows, so recomputing the
    # predicted class inside the loop would repeat the same work thousands of times.
    predicted = probabilities.argmax(axis=1)

    if metric in scoring.COUNT_METRICS:

        def measure(rows: np.ndarray) -> float:
            return scoring.from_counts(labels[rows], predicted[rows], len(class_names))[metric]
    else:

        def measure(rows: np.ndarray) -> float:
            return scoring.score(labels[rows], probabilities[rows], class_names)[metric]

    all_rows = np.arange(len(predictions))
    observed = measure(all_rows)
    rng = np.random.default_rng(seed)

    estimates = []
    for _ in range(resamples):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in drawn])
        estimates.append(measure(rows))

    low, high = np.percentile(estimates, _percentile_bounds(confidence))
    return Interval(
        estimate=float(observed), low=float(low), high=float(high), confidence=confidence
    )


def paired_difference(
    left: pd.Series,
    right: pd.Series,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    folds_per_repeat: int | None = None,
) -> PairedComparison:
    """Compare a model with its control on the folds they shared.

    Both series are indexed by fold, or by repeat and fold. The indices must match:
    comparing a model scored on one set of folds against a control scored on another
    would produce a number that looks like a margin and is not one.

    The test is the corrected resampled one rather than a plain paired t test, because
    folds of a cross validation are not independent of each other. On XGBoost against
    the metadata control the difference between the two is p = 0.00008 against
    p = 0.25, which is the difference between a finding and nothing of the kind.

    A p value from here is uncorrected for the number of comparisons the project
    reports. ``benjamini_hochberg`` does that across all of them at once.
    """
    if not left.index.equals(right.index):
        raise ComparisonError(
            "paired comparison needs the same folds on both sides; "
            f"got {list(left.index)} against {list(right.index)}"
        )
    if len(left) < 2:
        raise ComparisonError(f"cannot compare across {len(left)} fold")

    differences = (left - right).to_numpy(dtype=float)
    mean = float(differences.mean())

    if np.all(differences == differences[0]):
        # No spread at all, so a t test is undefined and the interval is the point.
        # Two folds that happen to differ by the same amount are not proof of
        # anything, so this reports no p value rather than zero. The test is on exact
        # equality, because a tolerance would send a merely narrow set of differences
        # down here and hide a result the t path can measure.
        return PairedComparison(
            difference=mean,
            low=mean,
            high=mean,
            p_value=1.0 if mean == 0 else float("nan"),
            folds_agreeing=len(differences) if mean != 0 else 0,
            n_folds=len(differences),
        )

    per_repeat = folds_per_repeat or _folds_per_repeat(left.index)
    error = np.sqrt(_variance_of_the_mean(differences, per_repeat))

    degrees = len(differences) - 1
    statistic = mean / error
    p_value = 2.0 * stats.t.sf(abs(statistic), degrees)
    half_width = stats.t.ppf(0.5 + confidence / 2.0, degrees) * error
    agreeing = int(np.sum(np.sign(differences) == np.sign(mean))) if mean != 0 else 0

    return PairedComparison(
        difference=mean,
        low=float(mean - half_width),
        high=float(mean + half_width),
        p_value=float(p_value),
        folds_agreeing=agreeing,
        n_folds=len(differences),
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjusted p values controlling the false discovery rate across a set of tests.

    Every margin this project publishes is a test, and printing two dozen of them
    beside each other at 0.05 apiece expects more than one to clear the bar carrying
    nothing. The step up procedure asks what the individual p values cannot: of the
    comparisons called resolved, how many are noise.

    This is deliberately less severe than dividing the threshold by the number of
    tests. Controlling the family wise error rate would ask that not one of two dozen
    comparisons is a false positive, which is a stricter question than this project
    needs to answer, and it would discard true findings to get there.

    A NaN carries through. A comparison whose folds all differed by the same amount
    has no p value, so it cannot be rejected and it does not lengthen the ranking for
    everything else either.
    """
    values = np.asarray(p_values, dtype=float)
    tested = ~np.isnan(values)
    adjusted = np.full(values.shape, np.nan)

    count = int(tested.sum())
    if count == 0:
        return adjusted

    ranked = values[tested]
    order = np.argsort(ranked, kind="stable")
    scaled = ranked[order] * count / np.arange(1, count + 1)

    # Walking back from the least significant keeps the result monotone, so a
    # comparison never lands below one that started out stronger than it.
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]

    result = np.empty(count)
    result[order] = np.minimum(monotone, 1.0)
    adjusted[tested] = result
    return adjusted


def shared_folds(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The splits both models were actually scored on.

    A network is fitted on one split while its control also ran nine further
    repeats, so only part of the control's scores has a partner. Narrowing to the
    overlap is what makes those two comparable, and the number of pairs left is
    reported so a five fold comparison is never read as a fifty fold one.
    """
    common = left.index.intersection(right.index)
    if len(common) < 2:
        raise ComparisonError(
            f"only {len(common)} split is shared between these two; nothing can be paired"
        )
    return left.loc[common].sort_index(), right.loc[common].sort_index()


def fold_scores(fold_metrics: pd.DataFrame, metric: str = "macro_f1") -> pd.Series:
    """One score per fold, indexed by fold, ready to be paired.

    The pairing is over the combination of the two: repeat three fold two of a model
    belongs with repeat three fold two of its control, because those are the same
    split. A run from before repeats existed is read as repeat zero, which is what it
    is, so it still pairs with the first repeat of a run that has ten.
    """
    if metric not in fold_metrics.columns:
        raise ComparisonError(f"no {metric} column in the fold metrics")
    if "fold" not in fold_metrics.columns:
        raise ComparisonError("fold metrics carry no fold column")
    if "repeat" not in fold_metrics.columns:
        fold_metrics = fold_metrics.assign(repeat=0)
    return fold_metrics.set_index(["repeat", "fold"])[metric].sort_index()
