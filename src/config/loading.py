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
    LOG_COMPRESSION,
    AudioConfig,
    Config,
    ConfigError,
    DatasetConfig,
    EncoderConfig,
    PathsConfig,
    PipelineConfig,
    SpectrogramConfig,
    SplitConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require(
    mapping: dict[str, Any],
    section: str,
    allowed: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    """A section that must be present, with every key in ``allowed`` supplied.

    ``optional`` names keys the section may carry and need not. They are accepted
    rather than refused as unknown, and their absence is not an error.
    """
    if section not in mapping:
        raise ConfigError(f"missing required config section: {section}")
    block = mapping[section]
    if not isinstance(block, dict):
        raise ConfigError(f"config section {section} must be a mapping")
    unknown = set(block) - allowed - (optional or set())
    if unknown:
        raise ConfigError(f"unknown keys in {section}: {sorted(unknown)}")
    missing = allowed - set(block)
    if missing:
        raise ConfigError(f"missing keys in {section}: {sorted(missing)}")
    return block


# Every section a config file may carry. A required one that goes missing is caught
# where it is read; an optional one that is misspelled was not caught anywhere.
SECTIONS = frozenset(
    {
        "extends",
        "name",
        "dataset",
        "audio",
        "spectrogram",
        "split",
        "paths",
        "pipeline",
        "encoder",
    }
)


def _check_sections(raw: dict[str, Any], source: Path) -> None:
    """Refuse a top level key this loader does not know.

    Keys inside a section were already checked and the section names were not, so a
    misspelled optional section fell through to its defaults in silence. That is not
    cosmetic here. Writing ``pipelines:`` instead of ``pipeline:`` leaves
    ``PipelineConfig.models`` as ``None``, which allows every model, so ``wide.yaml``
    would train the two networks it exists to exclude and rewrite a committed report.
    Writing ``encodder:`` leaves the checkpoint empty, and the encoder then runs with
    randomly initialised weights behind a log warning, producing embeddings that mean
    nothing while every stage downstream carries on.
    """
    unknown = set(raw) - SECTIONS
    if unknown:
        raise ConfigError(
            f"unknown top level sections in {source.name}: {sorted(unknown)}; "
            f"expected some of {sorted(SECTIONS)}"
        )


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


def _encoder(block: dict[str, Any]) -> EncoderConfig:
    """The encoder section, defaulted field by field so a partial block is legal.

    A config that names nothing gets an untrained stand in, which is what the
    synthetic corpus in the tests uses to exercise extraction without a download.
    """
    default = EncoderConfig()
    return EncoderConfig(
        architecture=str(block.get("architecture", default.architecture)),
        checkpoint=str(block.get("checkpoint", default.checkpoint)),
        url=str(block.get("url", default.url)),
        sha256=str(block.get("sha256", default.sha256)),
        sample_rate=int(block.get("sample_rate", default.sample_rate)),
        layer=int(block.get("layer", default.layer)),
        pooling=str(block.get("pooling", default.pooling)),
        embedding_dim=int(block.get("embedding_dim", default.embedding_dim)),
        batch_size=int(block.get("batch_size", default.batch_size)),
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read(source: Path) -> dict[str, Any]:
    if not source.exists():
        raise ConfigError(f"config file not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {source} must contain a top level mapping")
    return raw


def _overlay(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``override`` on top of ``base``, one level down.

    Sections merge key by key so a variant states only what differs, and a list
    replaces rather than appends: a species set or a model roster is the whole answer
    to its question, and extending one would give a variant every species its parent
    has plus its own, which is never what a narrower variant means.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = {**current, **value}
        else:
            merged[key] = value
    return merged


def _merged(source: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """One config, with whatever it extends underneath it.

    Four of the five experiment configs differed from ``base.yaml`` by nine to fourteen
    lines of forty four, and the rest was copied. That is not only length: the archive
    checksum was pinned in five files and the encoder block written out five times, so
    correcting either meant five correct edits and there was nothing to catch four.

    ``extends`` is resolved against the extending file's own directory, so a config
    tree can be moved without rewriting the links inside it.
    """
    if source in seen:
        chain = " -> ".join(path.name for path in (*seen, source))
        raise ConfigError(f"config extends itself: {chain}")

    raw = _read(source)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _overlay(_merged((source.parent / str(parent)).resolve(), (*seen, source)), raw)


def load_config(path: str | Path) -> Config:
    """Read a YAML config, validate it, and return the typed object."""
    source = Path(path)
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()

    raw = _merged(source)
    if "name" not in raw:
        raise ConfigError("config must define a top level 'name'")
    _check_sections(raw, source)

    dataset_block = _require(
        raw,
        "dataset",
        {"archive_url", "zip_name", "archive_sha256", "archive_root", "species"},
        optional={"corpus"},
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
    spec_block = _require(
        raw,
        "spectrogram",
        {"n_fft", "hop_length", "n_mels", "fmin", "fmax"},
        optional={"compression"},
    )
    split_block = _require(raw, "split", {"n_folds", "seed", "tape_id_length", "group_column"})
    paths_block = _require(raw, "paths", {"raw", "metadata", "processed", "reports"})
    pipeline_block = _optional(raw, "pipeline", {"models", "call_types"})
    encoder_block = _optional(
        raw,
        "encoder",
        {
            "architecture",
            "checkpoint",
            "url",
            "sha256",
            "sample_rate",
            "layer",
            "pooling",
            "embedding_dim",
            "batch_size",
        },
    )

    return Config(
        name=str(raw["name"]),
        dataset=DatasetConfig(
            archive_url=str(dataset_block["archive_url"]),
            zip_name=str(dataset_block["zip_name"]),
            archive_sha256=str(dataset_block["archive_sha256"]).lower(),
            archive_root=str(dataset_block["archive_root"]),
            species=tuple(str(s) for s in dataset_block["species"]),
            corpus=str(dataset_block.get("corpus", "")),
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
            compression=str(spec_block.get("compression", LOG_COMPRESSION)),
        ),
        split=SplitConfig(
            n_folds=int(split_block["n_folds"]),
            seed=int(split_block["seed"]),
            tape_id_length=int(split_block["tape_id_length"]),
            group_column=str(split_block["group_column"]),
        ),
        encoder=_encoder(encoder_block),
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


def experiment_configs() -> tuple[Path, ...]:
    """Every configuration that describes a corpus, discovered rather than listed.

    ``configs/`` also holds model hyperparameter files and the call type vocabulary,
    which are not experiments and do not load through ``load_config``. What marks an
    experiment is the corpus it describes or the one it inherits, rather than any one
    section, because a variant states only its differences.

    Listed instead of discovered, this went wrong twice. The multiplicity correction
    named five configurations in a tuple while the tests globbed the directory, so
    adding two left every published q value corrected across five sevenths of the
    comparisons with nothing raising.
    """
    found = []
    for path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and {"dataset", "extends"} & set(raw):
            found.append(path)
    return tuple(found)


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
