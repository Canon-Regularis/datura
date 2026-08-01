"""What the network actually uses: band occlusion, then Grad-CAM.

Both run against a saved fold checkpoint, so the model being explained is the one
that produced the reported score for that fold, on tapes it never saw.

Occlusion measures what the model loses when a band disappears; Grad-CAM shows
where the last convolutional stage responded. The first is a cost the model has to
pay, the second is a lead worth following, so they are reported together.

Usage:
    python -m src.evaluate.explain [--config configs/base.yaml] [--name cnn_small] [--fold 3]
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from src import cli
from src.config import Config
from src.data.splits import Fold, folds_for_index, rows_for_clips
from src.errors import DaturaError
from src.evaluate import plots
from src.evaluate.gradcam import GradCam
from src.evaluate.occlusion import band_occlusion
from src.features import registry as features
from src.features.source import CachedFeatureSource
from src.models import registry as models
from src.models.cnn import SpectrogramCNN
from src.models.registry import load_settings
from src.results import checkpoint_path, model_directory

logger = logging.getLogger(__name__)

EXAMPLES_PER_SPECIES = 2


class ExplainError(DaturaError):
    """Raised when there is no trained model to explain."""


def load_fold_model(
    cfg: Config, settings: dict, fold_index: int, n_classes: int, name: str
) -> SpectrogramCNN:
    checkpoint = checkpoint_path(cfg, name, fold_index)
    if not checkpoint.with_suffix(".pt").exists():
        raise ExplainError(
            f"no checkpoint at {checkpoint.with_suffix('.pt')}; "
            f"run python -m src.train.cnn --name {name} first"
        )
    return SpectrogramCNN.load(checkpoint, settings["train"], n_classes)


def sample_correct_windows(
    model: SpectrogramCNN,
    source: CachedFeatureSource,
    rows: np.ndarray,
    class_names: list[str],
    seed: int,
    per_species: int = EXAMPLES_PER_SPECIES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pick correctly classified windows, so the maps show evidence that worked."""
    rng = np.random.default_rng(seed)
    view = source.matrix(rows)
    predicted = model.predict_proba(view).argmax(axis=1)
    truth = source.index.iloc[rows]["label"].to_numpy()

    positions: list[int] = []
    labels: list[str] = []
    for label, name in enumerate(class_names):
        correct = np.flatnonzero((truth == label) & (predicted == label))
        pool = correct if len(correct) else np.flatnonzero(truth == label)
        if not len(pool):
            continue
        chosen = rng.choice(pool, size=min(per_species, len(pool)), replace=False)
        positions.extend(chosen.tolist())
        labels.extend(f"{name} (predicted {class_names[predicted[p]]})" for p in chosen.tolist())

    windows = view.take(np.asarray(positions, dtype=np.int64))
    heatmaps, _ = GradCam(model).heatmaps(windows)
    return windows, heatmaps, labels


def peak_frequencies(heatmaps: np.ndarray, frequencies: np.ndarray, labels: list[str]):
    """Where each map put most of its weight, in hertz."""
    return pd.DataFrame(
        {
            "example": labels,
            "peak_frequency_hz": [
                float(frequencies[int(np.argmax(h.mean(axis=1)))]) for h in heatmaps
            ],
        }
    )


def run(
    cfg: Config,
    settings: dict,
    fold: Fold,
    source: CachedFeatureSource,
    name: str,
) -> None:
    class_names = list(cfg.dataset.species)
    directory = model_directory(cfg, name)
    model = load_fold_model(cfg, settings, fold.index, len(class_names), name)
    rows = rows_for_clips(source.index, fold.test_clips)
    logger.info("explaining %s fold %d on %d held out windows", name, fold.index, len(rows))

    frequencies = features.build_extractor(features.LOGMEL, cfg).mel_frequencies()

    table = band_occlusion(model, source.matrix(rows), source.index, rows, class_names, frequencies)
    table.insert(0, "fold", fold.index)
    table.to_csv(directory / "occlusion.csv", index=False)
    plots.occlusion_profile(table, directory / "occlusion.png", class_names)
    logger.info(
        "\nMacro-F1 lost per masked band\n%s",
        table[["band_low_hz", "band_high_hz", "macro_f1", "macro_f1_drop"]]
        .round(3)
        .to_string(index=False),
    )

    windows, heatmaps, labels = sample_correct_windows(
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
    peak_frequencies(heatmaps, frequencies, labels).to_csv(
        directory / "gradcam_peaks.csv", index=False
    )
    logger.info("figures written to %s", directory)


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    cli.add_variant_name(parser, default="cnn_small")
    parser.add_argument("--fold", type=int, default=0, help="which fold's checkpoint to explain")
    args = parser.parse_args(argv)

    cfg = cli.prepare(args)
    spec = models.get(args.name)
    source = features.load_source(spec.source, cfg)
    folds = folds_for_index(source.index, cfg)
    if not 0 <= args.fold < len(folds):
        raise ExplainError(f"fold {args.fold} is outside 0..{len(folds) - 1}")

    run(cfg, load_settings(spec), folds[args.fold], source, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
