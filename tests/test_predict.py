"""Naming the species in one file, and declining to.

Two properties matter more than the rest.

A prediction here has to be the prediction the report published. If this path drifts
from the training path by so much as a normalisation step, the command answers a
different question from the one every number in the repository describes, and nothing
would say so. The round trip below is the check that caught that worry and settled it.

And the refusals have to be real. A recording below the band the model was trained on
must be turned away rather than upsampled, because the empty top of the spectrum is a
species label to a classifier and a silent wrong answer to a person.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import predict
from src.config import load_config
from src.data.manifest import load_manifest
from src.data.splits import folds_for_index
from src.evaluate import coverage
from src.results import predictions_path
from tests.helpers import needs

CONFIG = "configs/base.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def audio(cfg):
    root = cfg.paths.raw / cfg.dataset.archive_root
    needs(root, "run the pipeline that writes it")
    return root


def held_out_clip(cfg, species: str) -> tuple[str, str]:
    """One clip fold 0 never trained on, with the path it lives at."""
    manifest = load_manifest(cfg)
    index = manifest[["clip_id", "tape_id", "species", "label"]].copy()
    index["window_index"] = 0
    test = set(folds_for_index(index, cfg)[0].test_clips)

    rows = manifest[(manifest["species"] == species) & (manifest["clip_id"].isin(test))]
    if rows.empty:
        pytest.skip(f"no held out {species} clip")
    return rows.iloc[0]["clip_id"], rows.iloc[0]["relative_path"]


def test_a_prediction_reproduces_what_cross_validation_recorded(cfg, audio):
    """The property the whole command rests on.

    Fold 0's checkpoint scored these clips during cross validation and wrote the
    result. Running the same clip through here has to give the same probabilities, or
    the command is answering a different question from the report.
    """
    committed = pd.read_parquet(predictions_path(cfg, "xgboost"))
    columns = [c for c in committed.columns if c.startswith("p") and c[1:].isdigit()]

    for species in cfg.dataset.species:
        clip_id, relative = held_out_clip(cfg, species)
        row = committed[
            (committed["clip_id"] == clip_id)
            & (committed["repeat"] == 0)
            & (committed["fold"] == 0)
        ]
        if row.empty:
            continue

        fresh = predict.probabilities(cfg, audio / relative, ["xgboost"], 0)
        recorded = row.iloc[0][columns].to_numpy(dtype=float)
        assert fresh == pytest.approx(recorded, abs=1e-6), f"{clip_id} drifted from the report"


def test_a_recording_below_the_training_band_is_refused(cfg, audio, tmp_path):
    """Upsampling would add an empty high band, which a classifier reads as a species."""
    import soundfile as sf

    quiet = tmp_path / "too_slow.wav"
    sf.write(quiet, np.zeros(8000, dtype=np.float32), 4000)

    with pytest.raises(predict.CannotPredict, match="4000 Hz"):
        predict.probabilities(cfg, quiet, ["xgboost"], 0)


def test_a_recording_shorter_than_one_window_is_refused(cfg, tmp_path):
    import soundfile as sf

    sliver = tmp_path / "sliver.wav"
    sf.write(sliver, np.zeros(64, dtype=np.float32), cfg.audio.min_native_sample_rate)

    with pytest.raises(predict.CannotPredict):
        predict.probabilities(cfg, sliver, ["xgboost"], 0)


def test_the_threshold_is_read_from_each_model_rather_than_chosen(cfg):
    """One number cannot serve two models, and picking one by hand was a bug.

    A 0.6 cut declines 31% of the trees' answers and 5% of the network's, because the
    network is far more overconfident. Asking both to clear the same number left the
    shipped model calling a 99.7% wrong answer high confidence.
    """
    if coverage.for_model(cfg, "xgboost").empty:
        pytest.skip("no coverage table")

    trees = predict.threshold_for(predict.curve_for(cfg, "xgboost"))
    network = predict.threshold_for(predict.curve_for(cfg, "cnn_small"))

    assert trees is not None
    assert network is not None
    assert network > trees + 0.2, "these models need very different cut offs"


def test_a_higher_target_accuracy_declines_more(cfg):
    curve = predict.curve_for(cfg, "xgboost")
    if curve is None:
        pytest.skip("no coverage table")

    lenient = predict.threshold_for(curve, 0.85)
    strict = predict.threshold_for(curve, 0.92)
    assert strict >= lenient


def standing(cut_off=0.5, ceiling=0.95, matched=None):
    return predict.Standing(cut_off=cut_off, ceiling=ceiling, matched=matched)


def test_the_report_is_plain_ascii(cfg):
    """It crashed on the default Windows codepage, which is where it is mostly run."""
    text = predict.render(cfg, "xgboost", np.array([0.7, 0.2, 0.1]), standing())
    text.encode("cp1252")

    declined = predict.render(cfg, "xgboost", np.array([0.4, 0.35, 0.25]), standing())
    assert "UNCERTAIN" in declined
    declined.encode("cp1252")

    withheld = predict.render(cfg, "xgboost", np.array([0.9, 0.05, 0.05]), standing(cut_off=None))
    assert "WITHHELD" in withheld
    withheld.encode("cp1252")


def test_a_model_that_never_reaches_the_target_is_withheld_rather_than_offered(cfg):
    """The bug: two different absences were both spelled ``None``.

    Under ``configs/context.yaml`` no model exceeds 0.65 accuracy at any coverage, so
    no threshold earns the 90% target, and the old code read that as "no threshold to
    apply" and printed the answer with ``Confidence : HIGH``. A model that cannot be
    right often enough at any coverage has to withhold, not answer freely.
    """
    confident = np.array([0.97, 0.02, 0.01])

    never = predict.render(cfg, "xgboost", confident, standing(cut_off=None, ceiling=0.495))
    assert "WITHHELD" in never
    assert "49.5%" in never, "and it says how far short the model falls"
    assert "HIGH" not in never

    unknown = predict.render(cfg, "xgboost", confident, standing(ceiling=None))
    assert "no coverage table" in unknown, "no curve at all is a different sentence"


def test_the_three_states_are_read_off_the_committed_curves(cfg):
    """Both branches exist in the artifacts, so neither is hypothetical."""
    from src.config import load_config

    if predict.curve_for(cfg, "xgboost") is None:
        pytest.skip("no coverage table")

    on_recordings = predict.standing(cfg, "xgboost", 0.85)
    assert on_recordings.has_curve
    assert on_recordings.reaches_target

    context = load_config("configs/context.yaml")
    if predict.curve_for(context, "xgboost") is None:
        pytest.skip("context_10k has no coverage table")

    on_places = predict.standing(context, "xgboost", 0.85)
    assert on_places.has_curve, "the curve exists"
    assert not on_places.reaches_target, "and nothing on it reaches 90%"
    assert on_places.ceiling < predict.TARGET_ACCURACY


def test_averaging_two_models_needs_both_of_them(cfg, audio):
    """And gives something between them rather than one of them."""
    needs(predictions_path(cfg, "probe"), "run the pipeline that writes it")

    _, relative = held_out_clip(cfg, "KillerWhale")
    trees = predict.probabilities(cfg, audio / relative, ["xgboost"], 0)
    both = predict.probabilities(cfg, audio / relative, ["xgboost", "probe"], 0)

    assert both.shape == trees.shape
    assert both.sum() == pytest.approx(1.0, abs=1e-6)
    assert not np.allclose(both, trees), "averaging changed nothing, so it did not happen"


def test_an_unknown_model_is_refused_by_name(cfg):
    from src.models.registry import UnknownModel

    with pytest.raises(UnknownModel):
        predict.probabilities(cfg, cfg.source, ["not_a_model"], 0)


def test_the_shipped_fold_is_committed():
    """Without it a fresh clone can score the committed predictions but not a wav file."""
    from src.config import PROJECT_ROOT

    shipped = PROJECT_ROOT / "data/metadata/report/base_10k/xgboost/checkpoints/fold0.json"
    assert shipped.exists(), "the model `python -m src.predict` defaults to is missing"
