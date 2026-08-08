"""That every committed figure is produced by something and read by somebody.

Two ways a figure goes wrong here, and both had happened.

It stops being produced. ``ambiguity_breakdown.png`` sat in two report directories
for months after the code moved to one chart per giveaway field. The reproduce job
could not catch it, because matplotlib output is not byte stable across versions and
that job excludes every PNG from its diff.

Or it is produced and never shown. Notebooks are where a figure is read, and one that
appears in no notebook is a file nobody has looked at since it was written.

References are matched by name. A notebook writing ``f"confusion_{name}.png"`` covers
every confusion figure, which is the point of the loop it sits in, so a template
matches anything in the position of its placeholder. A reference that is nothing but
a placeholder would match every figure ever committed and assert nothing, so it is
refused rather than counted.
"""

from __future__ import annotations

import json
import re

import pytest

from src.config import PROJECT_ROOT, load_config
from src.evaluate import families, figures

REPORTS = PROJECT_ROOT / "data" / "metadata" / "report"
NOTEBOOKS = PROJECT_ROOT / "experiments"
CONFIG_FILE = {"base_10k": "base.yaml", "base_5k": "base_5k.yaml", "wide_10k": "wide.yaml"}

TOKEN = re.compile(r"[\w{}.\-]+\.png")
PLACEHOLDER = re.compile(r"\{[^}]*\}")


def committed() -> list[str]:
    if not REPORTS.exists():
        pytest.skip("report artifacts absent; run python -m src.evaluate.report first")
    return sorted(path.relative_to(REPORTS).as_posix() for path in REPORTS.rglob("*.png"))


def references() -> dict[str, set[str]]:
    """Every figure name mentioned in a notebook's source, by notebook."""
    found = {}
    for notebook in sorted(NOTEBOOKS.glob("*.ipynb")):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        names: set[str] = set()
        for cell in cells:
            names |= set(TOKEN.findall("".join(cell["source"])))
        found[notebook.stem] = names
    return found


def covers(reference: str, filename: str) -> bool:
    """Whether one reference names this figure, treating a placeholder as a wildcard."""
    pattern = ".+".join(re.escape(part) for part in PLACEHOLDER.split(reference))
    return re.fullmatch(pattern, filename) is not None


def test_a_reference_that_is_only_a_placeholder_is_refused():
    """It would match every figure in the tree and assert nothing about any of them."""
    for reference in {name for names in references().values() for name in names}:
        stripped = PLACEHOLDER.sub("", reference)
        assert stripped != ".png", f"{reference} asserts nothing"


def test_every_committed_figure_is_shown_in_a_notebook():
    everywhere = {name for names in references().values() for name in names}
    unseen = [
        figure
        for figure in committed()
        if not any(covers(reference, figure.rsplit("/", 1)[-1]) for reference in everywhere)
    ]
    assert not unseen, f"{len(unseen)} committed figures appear in no notebook: {unseen}"


# Figures a notebook draws for itself into a scratch directory rather than reading
# from the report tree. Notebook 01 plots the sample rate profile that way, because
# nothing in src writes it and a notebook may not add files to a tree the reproduce
# job diffs.
SELF_DRAWN = {"sample_rates.png"}


def test_every_literal_reference_names_a_figure_that_exists():
    """Catches a renamed figure, which a notebook would otherwise silently skip."""
    names = {figure.rsplit("/", 1)[-1] for figure in committed()} | SELF_DRAWN
    missing = {}
    for notebook, referenced in references().items():
        literal = {name for name in referenced if not PLACEHOLDER.search(name)}
        absent = sorted(name for name in literal if name not in names)
        if absent:
            missing[notebook] = absent
    assert not missing, f"notebooks name figures that are not in the report tree: {missing}"


def test_a_self_drawn_figure_is_not_committed():
    """If one of these ever lands in the report tree, nothing would regenerate it."""
    stray = [figure for figure in committed() if figure.rsplit("/", 1)[-1] in SELF_DRAWN]
    assert not stray, f"{stray} is drawn by a notebook and has no producer in src"


@pytest.mark.parametrize("name", sorted(CONFIG_FILE))
def test_every_figure_at_a_config_root_is_one_a_rebuild_writes(name, monkeypatch):
    """A figure nothing regenerates is a file the reproduce job cannot vouch for."""
    directory = REPORTS / name
    if not (directory / "comparison.csv").exists():
        pytest.skip(f"{name} has no report; run python -m src.evaluate.report first")

    import pandas as pd

    drawn: list = []
    for helper in ("model_comparison", "per_class_recall", "confusion_heatmap"):
        monkeypatch.setattr(figures.plots, helper, lambda *a, **k: drawn.append(a[1]) or a[1])
    monkeypatch.setattr(
        figures.plots, "ambiguity_comparison", lambda *a, **k: drawn.append(a[1]) or a[1]
    )
    monkeypatch.setattr(
        figures,
        "_per_model_drawing",
        lambda cfg, out: {
            "feature_importance.csv": lambda p, n: drawn.append(
                out / f"feature_importance_{n}.png"
            ),
            "history.csv": lambda p, n: drawn.append(out / f"training_history_{n}.png"),
            "occlusion.csv": lambda p, n: drawn.append(out / "occlusion.png"),
        },
    )

    cfg = load_config(f"configs/{CONFIG_FILE[name]}")
    family = next(f for f in families.discover(cfg) if f.key == families.SPECIES_FAMILY)
    figures.draw_all(
        cfg,
        family,
        pd.read_csv(directory / "comparison.csv"),
        pd.read_csv(directory / "ambiguity_breakdown.csv"),
    )

    produced = {path.name for path in drawn if path is not None}
    on_disk = {path.name for path in directory.glob("*.png")}
    assert on_disk <= produced, f"{sorted(on_disk - produced)} in {name} is regenerated by nothing"
