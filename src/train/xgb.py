"""Train the gradient boosted baselines: acoustic features, and the metadata control.

Both run here because they use the same estimator. Any gap between them comes from
the features rather than from the learner, which is exactly the comparison the
control is for.

Usage:
    python -m src.train.xgb [--config configs/base.yaml] [--model-config configs/xgb.yaml]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import Config, load_config, load_yaml
from src.data.splits import Fold, clips_from_index, fold_summary, make_folds
from src.features import cache
from src.features.extract import build_extractor
from src.features.source import CachedFeatureSource, FeatureSource, MetadataFeatureSource
from src.models.gbdt import GradientBoostedTrees
from src.models.metadata_control import build_metadata_control
from src.train.crossval import run_cross_validation, save_result


def load_acoustic_source(cfg: Config) -> CachedFeatureSource:
    store = cache.load_cached(cfg, "acoustic")
    names = build_extractor("acoustic", cfg).feature_names()
    return CachedFeatureSource(store, name="acoustic", feature_names=names)


def build_folds(cfg: Config, source: FeatureSource) -> list[Fold]:
    clips = clips_from_index(source.index)
    folds = make_folds(clips, cfg)
    summary = fold_summary(clips, folds)
    summary.to_csv(cfg.paths.reports / f"fold_summary_{cfg.name}.csv", index=False)
    print("\nTapes and clips per fold")
    print(
        summary[summary["part"] == "test"]
        .pivot(index="fold", columns="species", values="tapes")
        .to_string()
    )
    return folds


def _train_one(
    cfg: Config,
    source: FeatureSource,
    folds: list[Fold],
    model_name: str,
    factory,
    capture_importance: bool,
) -> None:
    print(f"\n{model_name} on {source.name} features")

    def hook(_: int, model: GradientBoostedTrees) -> dict[str, pd.DataFrame]:
        if not capture_importance:
            return {}
        scores = model.feature_importance(source.feature_names())
        table = pd.DataFrame(
            sorted(scores.items(), key=lambda item: -item[1]), columns=["feature", "gain"]
        )
        return {"feature_importance": table.head(60)}

    result = run_cross_validation(cfg, source, folds, factory, model_name, fold_hook=hook)
    directory = save_result(cfg, result)
    print(result.headline())
    print(f"  written to {directory}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="dataset and audio settings")
    parser.add_argument("--model-config", default="configs/xgb.yaml", help="estimator settings")
    parser.add_argument("--skip-control", action="store_true", help="omit the metadata control")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure()
    (cfg.paths.reports / cfg.name).mkdir(parents=True, exist_ok=True)
    params = load_yaml(args.model_config)["model"]

    source = load_acoustic_source(cfg)
    folds = build_folds(cfg, source)

    _train_one(
        cfg,
        source,
        folds,
        "xgboost",
        lambda: GradientBoostedTrees(params),
        capture_importance=True,
    )

    if not args.skip_control:
        control_source = MetadataFeatureSource(source.index)
        _train_one(
            cfg,
            control_source,
            folds,
            "metadata",
            lambda: build_metadata_control(cfg.split.seed),
            capture_importance=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
