"""Configuration: the shapes, and the loader that fills them in.

Callers import from this package. ``sections`` holds the validated dataclasses and
``loading`` reads the YAML, so the domain model stays free of file handling.
"""

from __future__ import annotations

from src.config.loading import PROJECT_ROOT, load_config, load_yaml
from src.config.sections import (
    AudioConfig,
    Config,
    ConfigError,
    DatasetConfig,
    EncoderConfig,
    PathsConfig,
    SpectrogramConfig,
    SplitConfig,
)

__all__ = [
    "PROJECT_ROOT",
    "AudioConfig",
    "Config",
    "ConfigError",
    "DatasetConfig",
    "EncoderConfig",
    "PathsConfig",
    "SpectrogramConfig",
    "SplitConfig",
    "load_config",
    "load_yaml",
]
