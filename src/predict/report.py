"""The prediction a person reads.

Kept apart from the policy that decides it, so changing the wording cannot change the
rule and changing the rule cannot silently change what a reader is told. Every branch
here corresponds to one state of ``Standing`` and there is no branch for a state that
does not exist.
"""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.predict.policy import DEFAULT_FOLD, TARGET_ACCURACY, Standing

WIDTH = 44


def render(
    cfg: Config, name: str, scores: np.ndarray, verdict: Standing, fold: int = DEFAULT_FOLD
) -> str:
    """The whole report for one recording, ranked species first."""
    species = list(cfg.dataset.species)
    order = np.argsort(scores)[::-1]

    lines = ["", "Species prediction", "-" * WIDTH]
    lines += [f"  {species[i]:<22s} {scores[i] * 100:5.1f}%" for i in order]
    lines.append("")
    lines += _verdict(species[int(order[0])], float(scores[int(order[0])]), verdict)
    lines += ["", f"  model: {name}, fold {fold}, trained on {cfg.name}", ""]
    return "\n".join(lines)


def _verdict(best: str, confidence: float, verdict: Standing) -> list[str]:
    if verdict.has_curve and not verdict.reaches_target:
        return [
            "  Prediction : WITHHELD",
            f"  Confidence : this model never reaches {TARGET_ACCURACY:.0%} accuracy at any",
            f"               coverage. Its best is {verdict.ceiling:.1%}, so no threshold",
            "               makes an answer from it worth acting on.",
        ]

    cut_off = verdict.cut_off
    if cut_off is not None and confidence < cut_off:
        return [
            "  Prediction : UNCERTAIN",
            f"  Confidence : LOW, below the {cut_off:.3f} this model needs to be "
            f"{TARGET_ACCURACY:.0%} accurate",
            "",
            "  This recording should not be classified automatically. The model is not",
            "  separating the species here.",
        ]

    return [f"  Prediction : {best}", *_confidence(verdict)]


def _confidence(verdict: Standing) -> list[str]:
    matched = verdict.matched
    if matched is not None:
        return [
            f"  Confidence : {_word(matched['coverage'])}, and on held out recordings "
            f"this model was {matched['accuracy']:.1%} accurate",
            f"               on the most confident {matched['coverage']:.0%} of its predictions",
        ]
    if verdict.has_curve:
        return [
            "  Confidence : below the weakest band this curve measures, so how often it",
            "               is right at this confidence is unknown",
        ]
    return ["  Confidence : no coverage table for this configuration"]


def _word(coverage_level: float) -> str:
    if coverage_level <= 0.5:
        return "HIGH"
    return "MODERATE" if coverage_level <= 0.8 else "LOW"
