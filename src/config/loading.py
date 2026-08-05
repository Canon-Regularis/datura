"""Reading a configuration file into validated objects.

This is the only place a YAML file is opened. Unknown keys and missing keys are
both refused, because a typo that silently falls back to a default is the kind of
mistake that shows up as a puzzling result three hours later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config.sections import (
    AudioConfig,
    Config,
    ConfigError,
    DatasetConfig,
    PathsConfig,
    PipelineConfig,
    SpectrogramConfig,
    SplitConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require(mapping: dict[str, Any], section: str, allowed: set[str]) -> dict[str, Any]:
    if section not in mapping:
        raise ConfigError(f"missing required config section: {section}")
    block = mapping[section]
    if not isinstance(block, dict):
        raise ConfigError(f"config section {section} must be a mapping")
    unknown = set(block) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in {section}: {sorted(unknown)}")
    missing = allowed - set(block)
    if missing:
        raise ConfigError(f"missing keys in {section}: {sorted(missing)}")
    return block


def _optional(mapping: dict[str, Any], section: str, allowed: set[str]) -> dict[str, Any]:
    """A section a config may leave out entirely.

    Absent means the default, so the configs that predate the section need no edit.
    Present is validated exactly as a required one: an unknown key is refused rather
    than ignored, because a typo here would silently restore the behaviour the
    section exists to restrict.
    """
    if section not in mapping:
        return {}
    block = mapping[section]
    if not isinstance(block, dict):
        raise ConfigError(f"config section {section} must be a mapping")
    unknown = set(block) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in {section}: {sorted(unknown)}")
    return block


def _names(block: dict[str, Any], key: str) -> tuple[str, ...] | None:
    """A declared list of names, or None when the section leaves it out."""
    if key not in block:
        return None
    value = block[key]
    if not isinstance(value, list):
        raise ConfigError(f"pipeline.{key} must be a list of names")
    return tuple(str(name) for name in value)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> Config:
    """Read a YAML config, validate it, and return the typed object."""
    source = Path(path)
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()
    if not source.exists():
        raise ConfigError(f"config file not found: {source}")

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {source} must contain a top level mapping")
    if "name" not in raw:
        raise ConfigError("config must define a top level 'name'")

    dataset_block = _require(
        raw,
        "dataset",
        {"archive_url", "zip_name", "archive_sha256", "archive_root", "species"},
    )
    audio_block = _require(
        raw,
        "audio",
        {
            "target_sample_rate",
            "min_native_sample_rate",
            "window_seconds",
            "hop_seconds",
            "min_clip_seconds",
            "pad_mode",
            "max_windows_per_clip",
        },
    )
    spec_block = _require(raw, "spectrogram", {"n_fft", "hop_length", "n_mels", "fmin", "fmax"})
    split_block = _require(raw, "split", {"n_folds", "seed", "tape_id_length", "group_column"})
    paths_block = _require(raw, "paths", {"raw", "metadata", "processed", "reports"})
    pipeline_block = _optional(raw, "pipeline", {"models", "call_types"})

    return Config(
        name=str(raw["name"]),
        dataset=DatasetConfig(
            archive_url=str(dataset_block["archive_url"]),
            zip_name=str(dataset_block["zip_name"]),
            archive_sha256=str(dataset_block["archive_sha256"]).lower(),
            archive_root=str(dataset_block["archive_root"]),
            species=tuple(str(s) for s in dataset_block["species"]),
        ),
        audio=AudioConfig(
            target_sample_rate=int(audio_block["target_sample_rate"]),
            min_native_sample_rate=int(audio_block["min_native_sample_rate"]),
            window_seconds=float(audio_block["window_seconds"]),
            hop_seconds=float(audio_block["hop_seconds"]),
            min_clip_seconds=float(audio_block["min_clip_seconds"]),
            pad_mode=str(audio_block["pad_mode"]),
            max_windows_per_clip=int(audio_block["max_windows_per_clip"]),
        ),
        spectrogram=SpectrogramConfig(
            n_fft=int(spec_block["n_fft"]),
            hop_length=int(spec_block["hop_length"]),
            n_mels=int(spec_block["n_mels"]),
            fmin=float(spec_block["fmin"]),
            fmax=float(spec_block["fmax"]),
        ),
        split=SplitConfig(
            n_folds=int(split_block["n_folds"]),
            seed=int(split_block["seed"]),
            tape_id_length=int(split_block["tape_id_length"]),
            group_column=str(split_block["group_column"]),
        ),
        pipeline=PipelineConfig(
            models=_names(pipeline_block, "models"),
            call_types=_names(pipeline_block, "call_types"),
        ),
        paths=PathsConfig(
            raw=_resolve(paths_block["raw"]),
            metadata=_resolve(paths_block["metadata"]),
            processed=_resolve(paths_block["processed"]),
            reports=_resolve(paths_block["reports"]),
        ),
        source=source,
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a model hyperparameter file. Model configs stay plain dicts because
    their keys are passed straight through to the estimator they configure."""
    source = Path(path)
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()
    if not source.exists():
        raise ConfigError(f"config file not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config file {source} must contain a top level mapping")
    return data
