"""The whole pipeline, on audio this file generates.

Unit tests cover each stage in isolation. This one catches the failures they cannot
see: a manifest column the feature cache does not carry, a fold addressed by clip id
against an index addressed by row, a report reading a file the trainer never wrote.

No download, no GPU, about a minute on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import scoring
from src.config import Config, load_config
from src.data.audit import audit_tables
from src.data.manifest import build_manifest, load_manifest
from src.data.splits import (
    assert_no_group_leak,
    clips_from_index,
    make_folds,
    rows_for_clips,
)
from src.evaluate import report
from src.features import cache
from src.features import registry as features
from src.features.extract import extract
from src.features.source import CachedFeatureSource, MetadataFeatureSource
from src.models.gbdt import GradientBoostedTrees
from src.results import model_directory
from src.train.crossval import run_cross_validation, save_result
from tests.conftest import write_config

SPECIES = ("HumpbackWhale", "SpermWhale", "KillerWhale")
TAPES_PER_SPECIES = 4
CUTS_PER_TAPE = 3
CLIP_SECONDS = 2.5

FAST_GBDT = {
    "n_estimators": 40,
    "max_depth": 3,
    "learning_rate": 0.2,
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": 0,
}


def _signal(species: str, rate: int, seed: int) -> np.ndarray:
    """One acoustic regime per species, roughly mirroring the real ones."""
    rng = np.random.default_rng(seed)
    n = int(CLIP_SECONDS * rate)
    t = np.arange(n) / rate
    noise = 0.02 * rng.standard_normal(n)

    if species == "HumpbackWhale":
        # Slow frequency modulated tone low in the band.
        sweep = 200 + 300 * np.sin(2 * np.pi * 0.7 * t)
        return (0.5 * np.sin(2 * np.pi * np.cumsum(sweep) / rate) + noise).astype(np.float32)

    if species == "SpermWhale":
        # Broadband impulses at a steady interval between clicks.
        signal = noise
        for start in range(0, n, int(0.25 * rate)):
            width = max(int(0.002 * rate), 4)
            signal[start : start + width] += rng.standard_normal(min(width, n - start))
        return signal.astype(np.float32)

    # Killer whale: a high whistle with a harmonic.
    whistle = 2600 + 700 * np.sin(2 * np.pi * 1.3 * t)
    phase = 2 * np.pi * np.cumsum(whistle) / rate
    return (0.4 * np.sin(phase) + 0.2 * np.sin(2 * phase) + noise).astype(np.float32)


def _write_tree(root: Path) -> None:
    """A miniature of the real layout: <Species>/<Year>/<8 char clip id>.wav."""
    rates = (16000, 22050, 32000, 16000)
    for species_index, species in enumerate(SPECIES):
        for tape in range(TAPES_PER_SPECIES):
            tape_id = f"{species_index}{tape:04d}"
            year = 1960 + tape
            for cut in range(CUTS_PER_TAPE):
                rate = rates[tape]
                directory = root / species / str(year)
                directory.mkdir(parents=True, exist_ok=True)
                seed = species_index * 100 + tape * 10 + cut
                sf.write(
                    directory / f"{tape_id}{cut:03d}.wav",
                    _signal(species, rate, seed),
                    rate,
                    subtype="PCM_16",
                )

    # One tape recorded below the target rate. The filter has to drop it.
    directory = root / SPECIES[0] / "1999"
    directory.mkdir(parents=True, exist_ok=True)
    for cut in range(CUTS_PER_TAPE):
        sf.write(
            directory / f"09999{cut:03d}.wav",
            _signal(SPECIES[0], 8000, 900 + cut),
            8000,
            subtype="PCM_16",
        )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Config:
    directory = tmp_path_factory.mktemp("e2e")
    path = write_config(
        directory,
        name="e2e",
        audio={"window_seconds": 1.0, "hop_seconds": 0.5, "max_windows_per_clip": 4},
        spectrogram={"n_fft": 256, "hop_length": 64, "n_mels": 32, "fmin": 50, "fmax": 4900},
        split={"n_folds": 2, "seed": 7, "tape_id_length": 5},
    )
    cfg = load_config(path)
    cfg.paths.ensure()
    _write_tree(cfg.paths.raw / cfg.dataset.archive_root)
    return cfg


@pytest.fixture(scope="module")
def manifest(dataset: Config):
    frame = build_manifest(dataset)
    frame.to_parquet(dataset.paths.metadata / f"manifest_{dataset.name}.parquet", index=False)
    return frame


@pytest.mark.slow
def test_manifest_drops_the_narrowband_tape(dataset, manifest):
    expected = len(SPECIES) * TAPES_PER_SPECIES * CUTS_PER_TAPE + CUTS_PER_TAPE
    assert len(manifest) == expected

    dropped = manifest[~manifest["keep"]]
    assert set(dropped["tape_id"]) == {"09999"}
    assert set(dropped["drop_reason"]) == {"native_rate_below_target"}

    kept = manifest[manifest["keep"]]
    assert kept.groupby("species")["tape_id"].nunique().to_dict() == dict.fromkeys(
        SPECIES, TAPES_PER_SPECIES
    )
    assert set(audit_tables(manifest)) >= {"audit_coverage", "audit_sample_rates"}


@pytest.mark.slow
def test_features_cover_every_kept_clip(dataset, manifest):
    kept = load_manifest(dataset, kept_only=True)
    for kind in features.kinds():
        extractor = features.build_extractor(kind, dataset)
        store = extract(dataset, extractor, kept)

        assert len(store.index) == len(store.features)
        assert store.index["clip_id"].nunique() == len(kept)
        assert store.index.groupby("clip_id").size().max() <= dataset.audio.max_windows_per_clip
        assert np.isfinite(np.asarray(store.features[:], dtype=np.float32)).all()
        assert features.cache_exists(kind, dataset)

    logmel = cache.load_cached(dataset, features.build_extractor(features.LOGMEL, dataset))
    extractor = features.build_extractor(features.LOGMEL, dataset)
    assert logmel.features.shape[1:] == extractor.output_shape(dataset.audio.window_samples)


@pytest.mark.slow
def test_cross_validation_produces_a_report(dataset, manifest):
    source = CachedFeatureSource(
        cache.load_cached(dataset, features.build_extractor(features.ACOUSTIC, dataset)),
        name="acoustic",
        feature_names=features.build_extractor(features.ACOUSTIC, dataset).feature_names(),
    )
    clips = clips_from_index(source.index)
    folds = make_folds(clips, dataset)
    assert_no_group_leak(clips, folds, dataset.split.group_column)

    for fold in folds:
        train_rows = rows_for_clips(source.index, fold.train_clips)
        test_rows = rows_for_clips(source.index, fold.test_clips)
        assert len(set(train_rows) & set(test_rows)) == 0

    audio_result = run_cross_validation(
        dataset, source, folds, lambda: GradientBoostedTrees(FAST_GBDT), "xgboost"
    )
    save_result(dataset, audio_result)

    control = MetadataFeatureSource(source.index)
    control_result = run_cross_validation(
        dataset, control, folds, lambda: GradientBoostedTrees(FAST_GBDT), "metadata"
    )
    save_result(dataset, control_result)

    # Three synthetic regimes this distinct should be easy. Anything near chance
    # means the labels and the features came apart somewhere upstream.
    assert audio_result.summary.set_index("metric").loc["macro_f1", "mean"] > 0.6

    report.build(dataset)
    directory = dataset.paths.reports / dataset.name
    for name in ("REPORT.md", "comparison.csv", "margin_over_control.csv", "provenance.json"):
        assert (directory / name).exists(), name
    for name in (
        "model_comparison.png",
        "per_class_recall.png",
        "ambiguity_native_sample_rate.png",
    ):
        assert (directory / name).stat().st_size > 0, name
    assert (model_directory(dataset, "xgboost") / "provenance.json").exists()


@pytest.mark.slow
def test_same_seed_gives_the_same_folds_and_scores(dataset, manifest):
    source = CachedFeatureSource(
        cache.load_cached(dataset, features.build_extractor(features.ACOUSTIC, dataset)),
        name="acoustic",
    )
    clips = clips_from_index(source.index)

    first = make_folds(clips, dataset)
    second = make_folds(clips, dataset)
    assert [f.test_clips for f in first] == [f.test_clips for f in second]

    scores = []
    for folds in (first, second):
        result = run_cross_validation(
            dataset,
            source,
            folds,
            lambda: GradientBoostedTrees(FAST_GBDT),
            "xgboost",
        )
        scores.append(result.clip_metrics["macro_f1"].to_numpy())
    np.testing.assert_allclose(scores[0], scores[1])


@pytest.mark.slow
def test_cnn_trains_and_explains_on_synthetic_audio(dataset, manifest):
    torch = pytest.importorskip("torch")
    from src.evaluate.gradcam import GradCam
    from src.evaluate.occlusion import band_occlusion
    from src.models.cnn import SpectrogramCNN

    source = CachedFeatureSource(
        cache.load_cached(dataset, features.build_extractor(features.LOGMEL, dataset)),
        name="logmel",
    )
    folds = make_folds(clips_from_index(source.index), dataset)
    settings = {
        "model": {"base_width": 4, "n_stages": 2, "blocks_per_stage": 1, "dropout": 0.1},
        "train": {
            "epochs": 1,
            "batch_size": 16,
            "lr": 0.01,
            "warmup_epochs": 0,
            "early_stopping_patience": 1,
            "seed": 0,
            "device": "cpu",
            "deterministic": True,
        },
        "augment": {"enabled": True, "probability": 1.0},
    }

    model = SpectrogramCNN(settings["model"], settings["train"], settings["augment"])
    index = source.index
    labels = index["label"].to_numpy()
    fold = folds[0]
    train_rows = rows_for_clips(index, fold.train_clips)
    validation_rows = rows_for_clips(index, fold.validation_clips)
    test_rows = rows_for_clips(index, fold.test_clips)

    from src.models.base import Batch

    model.fit(
        Batch(source.matrix(train_rows), labels[train_rows]),
        Batch(source.matrix(validation_rows), labels[validation_rows]),
        len(SPECIES),
    )
    probabilities = model.predict_proba(source.matrix(test_rows))
    assert probabilities.shape == (len(test_rows), len(SPECIES))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

    clips = scoring.aggregate_to_clips(index, test_rows, probabilities)
    assert len(clips) == len(fold.test_clips)

    frequencies = features.build_extractor(features.LOGMEL, dataset).mel_frequencies()
    occlusion = band_occlusion(
        model, source.matrix(test_rows), index, test_rows, list(SPECIES), frequencies, n_groups=4
    )
    assert len(occlusion) == 4
    assert occlusion["macro_f1_drop"].notna().all()

    windows = source.matrix(test_rows).take(np.arange(4))
    heatmaps, predicted = GradCam(model).heatmaps(windows)
    assert heatmaps.shape == windows.shape
    assert float(heatmaps.max()) <= 1.0 + 1e-6
    assert predicted.shape == (4,)
    assert isinstance(torch.tensor(0), torch.Tensor)
