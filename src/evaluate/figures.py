"""Which figures a report draws, and from what.

``src.evaluate.plots`` knows how to draw one chart. This module knows which charts a
configuration deserves and where the numbers behind them live. Keeping the two apart
means a new chart is a function in ``plots`` plus a line here, and neither of them
touches the document that links to the result.

Per model figures are drawn only when the file behind them exists, so a run that
trained the trees but not the network still produces a readable report rather than
failing on a missing learning curve.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.config import Config
from src.evaluate import families, plots, tables
from src.results import config_directory, model_directory

# Files a model may or may not write, and the figure each one becomes. The trees
# report which features carried the fit and the network reports its learning curve,
# so neither file is expected of both.
PER_MODEL = ("feature_importance.csv", "history.csv", "occlusion.csv")


def splits_behind(cfg: Config, name: str) -> int:
    """How many folds of how many repeats went into one model's pooled matrix.

    A confusion matrix pools every split, so a ten repeat run counts each clip ten
    times. The shares it is drawn from are unaffected and the raw counts printed
    beside them are not, so the title says how many splits are behind the number.
    """
    return len(pd.read_csv(model_directory(cfg, name) / "fold_metrics_clip.csv"))


def _per_model_drawing(cfg: Config, directory: Path) -> dict[str, Callable[[Path, str], Path]]:
    class_names = list(cfg.dataset.species)
    return {
        "feature_importance.csv": lambda path, name: plots.feature_importance(
            pd.read_csv(path), directory / f"feature_importance_{name}.png"
        ),
        "history.csv": lambda path, name: plots.training_history(
            pd.read_csv(path), directory / f"training_history_{name}.png"
        ),
        "occlusion.csv": lambda path, name: plots.occlusion_profile(
            pd.read_csv(path), directory / "occlusion.png", class_names
        ),
    }


def draw_all(
    cfg: Config,
    family: families.Family,
    comparison: pd.DataFrame,
    ambiguity: pd.DataFrame,
) -> list[Path]:
    """Every figure the species section links to, in the order it links to them.

    The family arrives whole rather than as a list of names, because the headline
    chart has to know which model is the floor. Reading it off a hardcoded name drew
    the reference line at the metadata control long after the project had established
    that the line belonged higher.
    """
    directory = config_directory(cfg)
    class_names = list(cfg.dataset.species)
    model_names = list(family.names)
    floor = families.strongest_floor(cfg, family) or family.control

    figures = [
        plots.model_comparison(
            tables.headline(comparison),
            directory / "model_comparison.png",
            floor=floor,
            silent=set(family.floors),
        ),
        plots.per_class_recall(comparison, directory / "per_class_recall.png", class_names),
    ]

    # One breakdown per giveaway. Native sample rate was the first field found to
    # hand over the species; the collection code is the second and the stronger.
    for field, rows in ambiguity.groupby("giveaway", sort=False):
        stem = str(field).replace(" ", "_")
        figures.append(
            plots.ambiguity_comparison(rows, directory / f"ambiguity_{stem}.png", str(field))
        )

    drawing = _per_model_drawing(cfg, directory)
    for name in model_names:
        source = model_directory(cfg, name)
        figures.append(
            plots.confusion_heatmap(
                pd.read_csv(source / "confusion.csv", index_col=0),
                directory / f"confusion_{name}.png",
                f"{name}, {splits_behind(cfg, name)} splits pooled",
            )
        )
        for filename in PER_MODEL:
            path = source / filename
            if path.exists():
                figures.append(drawing[filename](path, name))
    return figures
