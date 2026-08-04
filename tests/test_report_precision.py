"""Committed tables have to regenerate identically on another machine.

The reproduce job rebuilds every report from committed predictions and diffs the
result. That only works while the numbers are reproducible, and one of them was
not: scipy reaches the t distribution through the platform's libm, so the same
comparison wrote 0.010063775865938413 on Windows and 0.010063775865938426 on
Linux. Seventeen significant figures of a p value, disagreeing on the last two,
and the build failed on it.

The fix is to stop writing precision the design does not have. These tests hold
that line, because the natural thing to write is ``to_csv`` and the natural thing
to lose is the rounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate.report import CSV_DECIMALS, write_table


def test_a_table_is_written_at_the_declared_precision(tmp_path):
    frame = pd.DataFrame({"model": ["xgboost"], "p_value": [0.010063775865938413]})
    path = write_table(frame, tmp_path / "margins.csv")

    assert path.read_text(encoding="utf-8").strip().endswith("0.0100637759")


def test_two_values_a_last_digit_apart_are_written_the_same_way(tmp_path):
    """The exact failure: one machine's p value against another's."""
    windows = pd.DataFrame({"p_value": [0.010063775865938413]})
    linux = pd.DataFrame({"p_value": [0.010063775865938426]})

    assert write_table(windows, tmp_path / "a.csv").read_text(encoding="utf-8") == write_table(
        linux, tmp_path / "b.csv"
    ).read_text(encoding="utf-8")


def test_the_rounding_is_far_coarser_than_the_disagreement_and_far_finer_than_the_claim():
    """Ten decimals sits between what libm disagrees about and what anyone reads.

    Both bounds matter. Too fine and the build breaks again; too coarse and a
    margin or an interval loses a digit the report prints.
    """
    libm_disagreement = 1.3e-17
    finest_the_repo_ever_prints = 1e-4

    assert libm_disagreement * 100 < 10.0**-CSV_DECIMALS
    assert finest_the_repo_ever_prints > 10.0**-CSV_DECIMALS


def test_rounding_leaves_every_reported_figure_intact(tmp_path):
    """Nothing the report or the README quotes is changed by the rounding."""
    quoted = pd.DataFrame(
        {
            "margin": [-0.2407665, 0.1289812, -0.4961003],
            "low": [-0.3108712, -0.0142339, -0.5759821],
            "p_value": [9.4712e-09, 0.0763991, 1.2e-12],
        }
    )
    written = pd.read_csv(write_table(quoted, tmp_path / "m.csv"))

    assert np.allclose(written.to_numpy(), quoted.to_numpy(), atol=10.0**-CSV_DECIMALS)
    assert written.round(4).equals(quoted.round(4)), "the printed precision is untouched"


def test_text_columns_survive_the_rounding(tmp_path):
    frame = pd.DataFrame({"family": ["species"], "model": ["xgboost"], "folds": [50]})
    written = pd.read_csv(write_table(frame, tmp_path / "t.csv"))

    assert written["model"].iloc[0] == "xgboost"
    assert written["folds"].iloc[0] == 50
