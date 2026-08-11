"""Every figure the project draws.

Charts are grouped by what they are about: the dataset, the results, and what a
trained model relied on. They share one palette and one axis treatment, which live
in ``style``.

Callers import from this package rather than from the modules underneath, so a
chart can move between them without breaking anything.
"""

from __future__ import annotations

from src.evaluate.plots.audit import sample_rate_profile
from src.evaluate.plots.explain import gradcam_panel, occlusion_profile
from src.evaluate.plots.results import (
    ambiguity_comparison,
    confusion_heatmap,
    coverage_curve,
    feature_importance,
    model_comparison,
    per_class_recall,
    training_history,
)
from src.evaluate.plots.style import SEQUENTIAL, SERIES, species_colors

__all__ = [
    "SEQUENTIAL",
    "SERIES",
    "ambiguity_comparison",
    "confusion_heatmap",
    "coverage_curve",
    "feature_importance",
    "gradcam_panel",
    "model_comparison",
    "occlusion_profile",
    "per_class_recall",
    "sample_rate_profile",
    "species_colors",
    "training_history",
]
