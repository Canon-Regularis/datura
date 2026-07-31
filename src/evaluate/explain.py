"""What the CNN actually uses: band occlusion, then Grad-CAM.

Runs against a saved fold checkpoint, so the model being explained is the same one
that produced the reported score for that fold, evaluated on tapes it never saw.

Usage:
    python -m src.evaluate.explain [--config configs/base.yaml] [--fold 0]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import Config, load_config, load_yaml
from src.data.splits import Fold, clips_from_index, make_folds, rows_for_clips
from src.evaluate import plots
from src.evaluate.gradcam import GradCam
from src.evaluate.occlusion import band_occlusion
from src.features.extract import build_extractor
from src.features.source import CachedFeatureSource
from src.models.cnn import SpectrogramCNN
from src.train.cnn import load_spectrogram_source
from src.train.crossval import result_directory


class ExplainError(RuntimeError):
    pass


def _load_fold_model(
    cfg: Config, settings: dict, fold_index: int, n_classes: int, name: str
) -> SpectrogramCNN:
    checkpoint = result_directory(cfg, name) / "checkpoints" / f"fold{fold_index}"
    if not checkpoint.with_suffix(".pt").exists():
        raise ExplainError(
            f"no checkpoint at {checkpoint.with_suffix('.pt')}; run python -m src.train.cnn first"
        )
    return SpectrogramCNN.load(checkpoint, settings["train"], n_classes)


def _gradcam_examples(
    model: SpectrogramCNN,
    source: CachedFeatureSource,
    rows: np.ndarray,
    class_names: list[str],
    seed: int,
    per_species: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample correctly classified windows so the maps show working evidence."""
    rng = np.random.default_rng(seed)
    view = source.matrix(rows)
    predicted = model.predict_proba(view).argmax(axis=1)
    truth = source.index.iloc[rows]["label"].to_numpy()

    positions = []
    labels = []
    for label, name in enumerate(class_names):
        correct = np.flatnonzero((truth == label) & (predicted == label))
        pool = correct if len(correct) else np.flatnonzero(truth == label)
        if not len(pool):
            continue
        chosen = rng.choice(pool, size=min(per_species, len(pool)), replace=False)
        positions.extend(chosen.tolist())
        labels.extend(
            f"{name} (predicted {class_names[predicted[p]]})" for p in chosen.tolist()
        )

    windows = view.take(np.asarray(positions, dtype=np.int64))
    heatmaps, _ = GradCam(model).heatmaps(windows)
    return windows, heatmaps, labels


def run(
    cfg: Config,
    settings: dict,
    fold: Fold,
    source: CachedFeatureSource,
    name: str = "cnn",
) -> None:
    class_names = list(cfg.dataset.species)
    directory = result_directory(cfg, name)
    model = _load_fold_model(cfg, settings, fold.index, len(class_names), name)
    rows = rows_for_clips(source.index, fold.test_clips)
    print(f"explaining fold {fold.index} on {len(rows)} held-out windows")

    extractor = build_extractor("logmel", cfg)
    frequencies = extractor.mel_frequencies()

    table = band_occlusion(
        model, source.matrix(rows), source.index, rows, class_names, frequencies
    )
    table.insert(0, "fold", fold.index)
    table.to_csv(directory / "occlusion.csv", index=False)
    plots.occlusion_profile(table, directory / "occlusion.png", class_names)

    print("\nMacro-F1 lost per masked band")
    print(
        table[["band_low_hz", "band_high_hz", "macro_f1", "macro_f1_drop"]]
        .round(3)
        .to_string(index=False)
    )

    windows, heatmaps, labels = _gradcam_examples(
        model, source, rows, class_names, cfg.split.seed
    )
    plots.gradcam_panel(
        windows,
        heatmaps,
        labels,
        directory / "gradcam.png",
        frequencies=frequencies,
        seconds=cfg.audio.window_seconds,
    )

    attention = pd.DataFrame(
        {
            "example": labels,
            "peak_frequency_hz": [
                float(frequencies[int(np.argmax(h.mean(axis=1)))]) for h in heatmaps
            ],
        }
    )
    attention.to_csv(directory / "gradcam_peaks.csv", index=False)
    print(f"\nfigures written to {directory}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model-config", default="configs/cnn.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--name", default="cnn", help="which trained variant to explain")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    settings = load_yaml(args.model_config)
    source = load_spectrogram_source(cfg)
    folds = make_folds(clips_from_index(source.index), cfg)
    if not 0 <= args.fold < len(folds):
        raise ExplainError(f"fold {args.fold} is outside 0..{len(folds) - 1}")

    run(cfg, settings, folds[args.fold], source, name=args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
