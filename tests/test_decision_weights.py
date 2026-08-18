"""Moving the decision boundary, and where the move is allowed to be chosen.

Macro-F1 averages the per class scores, so a class the model under predicts costs as
much as a common class it gets right. Humpback is predicted at 0.727 precision against
0.608 recall, which is a boundary held back rather than a class the model cannot see,
and one multiplier per class is enough to move it.

The arithmetic is small. What matters is that the multipliers are fitted on rows the
model is not scored on, so the tests below are mostly about that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import scoring
from tests.helpers import SPECIES


def probabilities(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=float)


def test_a_balanced_model_is_left_alone():
    """Nothing to gain means nothing moves, so a run that needs no help reports none."""
    labels = np.array([0, 1, 2, 0, 1, 2])
    perfect = probabilities([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]] * 2)
    weights = scoring.fit_decision_weights(labels, perfect, list(SPECIES))

    np.testing.assert_allclose(weights, np.ones(3))


def test_an_under_predicted_class_gets_lifted():
    """The whole point. A class that never wins an argmax it nearly wins should win it.

    Six clips, two of each class, and the first class is second by a nose every time it
    should be first. Its recall is zero on the argmax and macro-F1 pays for it.
    """
    labels = np.array([0, 0, 1, 1, 2, 2])
    shy = probabilities(
        [
            [0.40, 0.45, 0.15],
            [0.40, 0.45, 0.15],
            [0.10, 0.80, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.10, 0.10, 0.80],
        ]
    )
    before = scoring.from_counts(labels, shy.argmax(axis=1), 3)["macro_f1"]
    weights = scoring.fit_decision_weights(labels, shy, list(SPECIES))
    after = scoring.from_counts(labels, (shy * weights).argmax(axis=1), 3)["macro_f1"]

    assert weights[0] > 1.0, "the shy class should be lifted"
    assert after > before, f"macro-F1 went {before:.3f} to {after:.3f}"


def test_the_probabilities_are_never_rewritten():
    """Only the decision moves.

    The coverage curve and the confident error rate both read these columns, and both
    are claims about how sure the model was rather than about what it answered. A
    weight that edited them would quietly restate every abstention number.
    """
    index = pd.DataFrame(
        {
            "clip_id": ["1000100", "1000101", "2000100"],
            "tape_id": ["10001", "10001", "20001"],
            "species": [SPECIES[0], SPECIES[0], SPECIES[1]],
            "label": [0, 0, 1],
        }
    )
    raw = probabilities([[0.4, 0.6, 0.0], [0.4, 0.6, 0.0], [0.1, 0.9, 0.0]])
    rows = np.arange(3)

    plain = scoring.aggregate_to_clips(index, rows, raw)
    weighted = scoring.aggregate_to_clips(index, rows, raw, weights=np.array([4.0, 1.0, 1.0]))

    columns = scoring.probability_columns(3)
    pd.testing.assert_frame_equal(plain[columns], weighted[columns])
    assert list(plain["prediction"]) == [1, 1, 1]
    assert list(weighted["prediction"]) == [0, 0, 1], "the boundary moved and nothing else"


def test_the_ranking_metrics_ignore_the_boundary():
    """Reweighting a decision does not change how well a model ordered the clips.

    So the AUC has to come from the probabilities either way. Reading it off the moved
    decision would make a calibration look like a better model.
    """
    labels = np.array([0, 0, 1, 1])
    raw = probabilities([[0.45, 0.55, 0.0], [0.45, 0.55, 0.0], [0.2, 0.8, 0.0], [0.3, 0.7, 0.0]])

    plain = scoring.score(labels, raw, list(SPECIES))
    moved = scoring.score(labels, raw, list(SPECIES), predictions=np.array([0, 0, 1, 1]))

    assert moved["macro_f1"] > plain["macro_f1"], "the decision improved"
    assert moved["roc_auc_ovr_macro"] == plain["roc_auc_ovr_macro"], "the ordering did not"


def test_the_weights_are_deterministic():
    """Two calls on one input give one answer, so a refit cannot move a published score.

    Coordinate ascent over a fixed grid rather than an optimiser with a starting point,
    for exactly this reason.
    """
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=200)
    raw = rng.dirichlet(np.ones(3), size=200)

    first = scoring.fit_decision_weights(labels, raw, list(SPECIES))
    second = scoring.fit_decision_weights(labels, raw, list(SPECIES))

    np.testing.assert_array_equal(first, second)
