"""Execute the notebooks against the committed report artifacts.

The notebooks are the human facing view of the results, so a rename in the report
layout that leaves them raising FileNotFoundError should fail the build rather than
wait to be discovered by whoever opens them next.

Nothing is written back. The tracked copies keep the outputs they were saved with.
"""

from __future__ import annotations

import pytest

nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient

from src.config import PROJECT_ROOT  # noqa: E402

NOTEBOOKS = sorted((PROJECT_ROOT / "experiments").glob("*.ipynb"))
REPORT = PROJECT_ROOT / "data" / "metadata" / "report" / "base_10k"


def test_notebooks_are_present():
    assert NOTEBOOKS, "no notebooks found under experiments/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_valid(path):
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path):
    if not (REPORT / "comparison.csv").exists():
        pytest.skip("report artifacts absent; run python -m src.pipeline first")

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT / "experiments")}},
    )
    client.execute()
