"""Train the log-mel CNN under the same folds as the baselines.

Usage:
    python -m src.train.cnn [--config configs/base.yaml] [--model-config configs/cnn.yaml]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import torch

from src.config import Config, load_config, load_yaml
from src.data.splits import Fold
from src.features import cache
from src.features.source import CachedFeatureSource
from src.models.cnn import SpectrogramCNN, resolve_device
from src.train.crossval import result_directory, run_cross_validation, save_result
from src.train.xgb import build_folds


def load_spectrogram_source(cfg: Config) -> CachedFeatureSource:
    return CachedFeatureSource(cache.load_cached(cfg, "logmel"), name="logmel")


def _describe_device(settings: dict) -> torch.device:
    device = resolve_device(str(settings.get("device", "auto")))
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"training on {properties.name}, {properties.total_memory / 1e9:.1f} GB")
    else:
        print(f"training on CPU with {torch.get_num_threads()} threads")
    return device


def train(
    cfg: Config,
    settings: dict,
    folds: list[Fold],
    source: CachedFeatureSource,
    name: str = "cnn",
) -> None:
    _describe_device(settings["train"])

    def build() -> SpectrogramCNN:
        return SpectrogramCNN(settings["model"], settings["train"], settings["augment"])

    histories: dict[int, pd.DataFrame] = {}

    def hook(fold_index: int, model: SpectrogramCNN) -> dict[str, pd.DataFrame]:
        histories[fold_index] = pd.DataFrame(model.history)
        checkpoint = result_directory(cfg, name) / "checkpoints" / f"fold{fold_index}"
        model.save(checkpoint)
        return {"history": histories[fold_index]}

    result = run_cross_validation(cfg, source, folds, build, name, fold_hook=hook)
    directory = save_result(cfg, result)
    print(result.headline())
    print(f"  written to {directory}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="dataset and audio settings")
    parser.add_argument("--model-config", default="configs/cnn.yaml", help="network settings")
    parser.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    parser.add_argument("--name", default="cnn", help="result directory name for this variant")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure()
    (cfg.paths.reports / cfg.name).mkdir(parents=True, exist_ok=True)

    settings = load_yaml(args.model_config)
    if args.epochs is not None:
        settings["train"]["epochs"] = args.epochs

    source = load_spectrogram_source(cfg)
    folds = build_folds(cfg, source)
    train(cfg, settings, folds, source, name=args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
