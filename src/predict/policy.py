"""Whether a model may answer at all, and what its answer is worth.

Every threshold here comes off a model's own measured coverage curve rather than being
chosen. The same probability means different things to different models: a 0.6 cut
declines 31.5% of the trees' answers and 5.3% of the network's, because the network is
far more overconfident. To reach 90% accuracy ``cnn_small`` needs 0.954 where
``xgboost`` needs 0.591. One number cannot serve both, and the one that was hardcoded
left the shipped model calling a 99.7% wrong answer HIGH confidence.

Abstention filters uncertainty, not error. It removes the ambiguous cases and leaves
the confident mistakes, so a model that is often wrong while sure stays wrong while
sure. That is why the shipped model is the one that is confidently wrong least often
rather than the one that scores best on any single fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import Config
from src.evaluate import coverage
from src.results import coverage_path

# The trees rather than the network. Pooled over all 41,600 held out predictions the
# network is wrong while above 90% confident on 8.62% of them, and the trees on 0.10%.
# Eighty five times fewer confident mistakes, and a higher score besides. Shipping the
# network was a mistake made because it was the only model that saved itself.
DEFAULT_MODEL = "xgboost"
DEFAULT_FOLD = 0

# What accuracy a prediction has to be worth before it is offered as one. The cut off
# probability is then read from the model's own coverage curve, because a fixed one is
# meaningless across models.
TARGET_ACCURACY = 0.90


@dataclass(frozen=True)
class Standing:
    """What a model's curve says about a prediction of a given confidence.

    Three states, and collapsing two of them is how this command came to print
    ``Confidence : HIGH`` beside a model that is right 49.5% of the time. There was one
    nullable float, and ``None`` meant both "this configuration has no curve" and "this
    model never reaches the target at any coverage". The second is a refusal and the
    first is an apology, and the code took the confident branch for both.
    """

    cut_off: float | None
    ceiling: float | None
    matched: pd.Series | None

    @property
    def has_curve(self) -> bool:
        """Whether anything is known about how often this model is right."""
        return self.ceiling is not None

    @property
    def reaches_target(self) -> bool:
        """Whether declining the least confident clips ever earns the target accuracy."""
        return self.cut_off is not None


def curve_for(cfg: Config, name: str) -> pd.DataFrame | None:
    """This model's measured accuracy against coverage, or nothing if it has none."""
    path = coverage_path(cfg)
    if not path.exists():
        return None
    table = pd.read_csv(path)
    rows = table[table["model"] == name]
    return rows if len(rows) else None


def threshold_for(curve: pd.DataFrame | None, target: float = TARGET_ACCURACY) -> float | None:
    """The probability below which this model should decline, to earn ``target``.

    Taken from the curve rather than chosen, so it adapts to how confident a given
    model happens to be. A network whose probabilities are all above 0.9 needs a far
    higher cut off than a tree whose probabilities spread out, and asking both to
    clear the same number gives one of them an abstention rule that never fires.

    ``None`` means no coverage level reaches the target, which is a model that should
    not be answering. Callers want ``standing``, which can tell that apart from a
    configuration that has no curve at all.
    """
    if curve is None:
        return None
    good = curve[curve["accuracy"] >= target].sort_values("coverage", ascending=False)
    return float(good.iloc[0]["threshold"]) if len(good) else None


def standing(
    cfg: Config, name: str, confidence: float, target: float = TARGET_ACCURACY
) -> Standing:
    """Everything this model's own curve says about a prediction of this confidence."""
    curve = curve_for(cfg, name)
    if curve is None:
        return Standing(cut_off=None, ceiling=None, matched=None)
    return Standing(
        cut_off=threshold_for(curve, target),
        ceiling=float(curve["accuracy"].max()),
        matched=coverage.band(curve, confidence),
    )


def band(cfg: Config, name: str, confidence: float) -> pd.Series | None:
    """What accuracy this confidence earned on held out recordings."""
    curve = curve_for(cfg, name)
    return coverage.band(curve, confidence) if curve is not None else None
