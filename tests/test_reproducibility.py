"""Which committed numbers can be regenerated, and which cannot.

This project argues that a number without its uncertainty is worth very little, so it
should be able to say which of its own numbers survive a rerun. Four of the six models
do and two do not, and until this file existed nothing said so anywhere.

The trees and the probe reproduce their committed predictions bit for bit. The two
networks do not: ``configs/cnn.yaml`` and ``configs/cnn_small.yaml`` set
``deterministic: false``, so cuDNN benchmarks its kernels once and keeps whichever was
quickest on the day. Refitting fold 0 of ``cnn_small`` with untouched code disagrees
with the committed predictions on 133 of 797 clips.

That was found the hard way. A refactor of the training loop was checked against the
committed predictions, the check failed, and the failure turned out to be the model
rather than the change. Had the probe not been reproducible, a correct commit would
have been thrown away on the strength of it.

Forcing determinism is not the answer and the reasoning is in the README's limits. It
costs about a third of the throughput, it does not hold across hardware, driver or
cuDNN version, and it would replace an honest source of variance with one arbitrary
draw presented as exact. What belongs here instead is a statement of where the line
falls, so nobody has to rediscover it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src import scoring
from src.config import PROJECT_ROOT, load_config
from src.models import registry as models
from src.results import clip_metrics_path, predictions_path
from tests.helpers import needs

CONFIG = "configs/base.yaml"

# Every model whose fit is a deterministic function of its inputs and seed. The trees
# pin their thread count for exactly this reason, and the probe is a linear map over
# cached vectors with no kernel choice to make.
REPRODUCIBLE = ("xgboost", "xgboost_centred", "metadata", "logbook", "probe")
NOT_REPRODUCIBLE = ("cnn", "cnn_small")


def settings_of(name: str) -> dict:
    spec = models.get(name)
    return yaml.safe_load((PROJECT_ROOT / spec.config_file).read_text(encoding="utf-8"))


def test_every_model_is_accounted_for():
    """So a seventh model cannot be added without deciding which side it falls on."""
    assert set(REPRODUCIBLE) | set(NOT_REPRODUCIBLE) == set(models.names())


def test_the_networks_declare_that_they_are_not_reproducible():
    """The claim in the README is tied to the config rather than asserted beside it."""
    for name in NOT_REPRODUCIBLE:
        assert settings_of(name)["train"]["deterministic"] is False, (
            f"{name} now asks for determinism, so the limits section is out of date"
        )


def test_the_trees_pin_their_thread_count():
    """XGBoost sums histograms in whatever order the threads finish.

    With ``n_jobs: -1`` a committed score would depend on the core count of the machine
    that wrote it, which is the same failure as the networks' and just as quiet.
    """
    for name in ("xgboost", "xgboost_centred", "metadata", "logbook"):
        assert settings_of(name)["model"]["n_jobs"] > 0


@pytest.mark.slow
def test_the_probe_reproduces_its_committed_predictions():
    """The strongest reproducibility claim the project makes, and the only one checked.

    One fold refitted from scratch has to give back exactly what is committed. If this
    ever fails, either the training path changed or a model that was deterministic
    stopped being so, and both are worth stopping for.
    """
    from src.data.splits import rows_for_clips
    from src.features import registry as features
    from src.models.base import Batch
    from src.models.registry import load_settings
    from src.train.folds import folds_for

    cfg = load_config(CONFIG)
    committed = predictions_path(cfg, "probe")
    needs(committed, "run the pipeline first")

    spec = models.get("probe")
    source = features.load_source(spec.source, cfg)
    index = source.index
    labels = index["label"].to_numpy()
    fold = folds_for(cfg, source)[0]

    train, validation, test = (
        rows_for_clips(index, fold.train_clips),
        rows_for_clips(index, fold.validation_clips),
        rows_for_clips(index, fold.test_clips),
    )
    model = spec.build(cfg, load_settings(spec))
    model.fit(
        Batch(source.matrix(train), labels[train]),
        Batch(source.matrix(validation), labels[validation]),
        len(cfg.dataset.species),
    )
    clips, _ = scoring.evaluate_clips(
        index, test, model.predict_proba(source.matrix(test)), list(cfg.dataset.species)
    )

    columns = scoring.probability_columns(len(cfg.dataset.species))
    published = pd.read_parquet(committed)
    fold_zero = published[(published["repeat"] == 0) & (published["fold"] == 0)]
    fold_zero = fold_zero.sort_values("clip_id")

    assert len(clips) == len(fold_zero)
    assert np.array_equal(clips[columns].to_numpy(), fold_zero[columns].to_numpy())


def test_the_published_network_spread_absorbs_its_own_reruns():
    """``cnn_small`` redraws the split ten times, so seed variance is inside its spread.

    ``cnn`` runs one split, so its five estimates carry the same run to run variance
    with nothing separating it from the difference between folds. That is why the
    limits section names it and not the other.
    """
    cfg = load_config(CONFIG)
    path = clip_metrics_path(cfg, "cnn_small")
    needs(path, "run the pipeline that writes it")

    folds = pd.read_csv(path)
    assert folds["repeat"].nunique() == 10, "ten redraws is what absorbs the variance"

    single = clip_metrics_path(cfg, "cnn")
    if single.exists():
        assert pd.read_csv(single)["repeat"].nunique() == 1, "and cnn does not redraw"


def test_the_limits_section_quotes_the_spread_the_folds_actually_show():
    """The README compares a rerun of one fold against the spread across all fifty.

    Only the second half of that comparison is committed here. The first, 0.517 to
    0.638 over seven refits of fold 0 on one seed, came from a measurement that writes
    nothing to the repository, so it is quoted rather than checked. This holds the
    number it is quoted against, which is the one that would move if the network were
    refitted.
    """
    cfg = load_config(CONFIG)
    path = clip_metrics_path(cfg, "cnn_small")
    needs(path, "run the pipeline that writes it")

    measured = pd.read_csv(path)["macro_f1"].std(ddof=1)
    quoted = 0.142
    assert measured == pytest.approx(quoted, abs=5e-4), (
        f"the limits section says {quoted} across the fifty splits and they give {measured:.4f}"
    )
