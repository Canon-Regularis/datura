"""Where results live on disk.

Every path under the report tree is built here. Training writes a checkpoint,
explainability reads it back, and the report walks the same directories: if any of
them spelled the layout out for itself, renaming a folder would break the others
silently.
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config

CHECKPOINTS = "checkpoints"
SUMMARY_FILE = "summary.csv"
PROVENANCE_FILE = "provenance.json"
PREDICTIONS_FILE = "clip_predictions.parquet"
REPORT_FILE = "REPORT.md"


def config_directory(cfg: Config) -> Path:
    """Everything produced under one configuration."""
    return cfg.paths.reports / cfg.name


def model_directory(cfg: Config, model_name: str) -> Path:
    """One trained model's metrics, predictions and figures."""
    return config_directory(cfg) / model_name


def checkpoint_path(cfg: Config, model_name: str, fold_index: int, repeat: int = 0) -> Path:
    """The weights saved for one fold of one repeat.

    Repeat zero keeps the plain name, so a single split writes exactly where it
    always did and the explainability tools keep finding it. The suffix is added by
    the writer.
    """
    stem = f"fold{fold_index}" if repeat == 0 else f"repeat{repeat}_fold{fold_index}"
    return model_directory(cfg, model_name) / CHECKPOINTS / stem


def summary_path(cfg: Config, model_name: str) -> Path:
    return model_directory(cfg, model_name) / SUMMARY_FILE


def predictions_path(cfg: Config, model_name: str) -> Path:
    return model_directory(cfg, model_name) / PREDICTIONS_FILE


def fold_summary_path(cfg: Config) -> Path:
    return cfg.paths.reports / f"fold_summary_{cfg.name}.csv"


def report_path(cfg: Config) -> Path:
    return config_directory(cfg) / REPORT_FILE


def has_results(cfg: Config, model_name: str) -> bool:
    """Whether a model has been trained under this configuration."""
    return summary_path(cfg, model_name).exists()


def ensure(cfg: Config) -> Path:
    """Create the report tree for this configuration and return its root."""
    directory = config_directory(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
