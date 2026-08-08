from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import Config, load_config

_TEMPLATE = {
    "name": "test",
    "dataset": {
        "archive_url": "https://example.invalid/archive.zip",
        "zip_name": "archive.zip",
        "archive_sha256": "0" * 64,
        "archive_root": "Watkins_Marine_Mammal_Sound_Database",
        "species": ["HumpbackWhale", "SpermWhale", "KillerWhale"],
    },
    "audio": {
        "target_sample_rate": 10000,
        "min_native_sample_rate": 10000,
        "window_seconds": 2.0,
        "hop_seconds": 1.0,
        "min_clip_seconds": 0.5,
        "pad_mode": "reflect",
        "max_windows_per_clip": 16,
    },
    "spectrogram": {
        "n_fft": 512,
        "hop_length": 64,
        "n_mels": 64,
        "fmin": 50,
        "fmax": 4900,
    },
    "split": {"n_folds": 5, "seed": 1234, "tape_id_length": 5, "group_column": "tape_id"},
    # A two layer encoder with no checkpoint, so every test that walks the extractors
    # exercises the real path offline in a second. The published configs name a 95
    # million parameter model and real weights, and a test asserts they do.
    "encoder": {
        "architecture": "wav2vec2_tiny",
        "checkpoint": "",
        "embedding_dim": 32,
        "layer": 1,
        "batch_size": 4,
    },
    "paths": {},
}


def write_config(directory: Path, **overrides) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        key: dict(value) if isinstance(value, dict) else value for key, value in _TEMPLATE.items()
    }
    payload["paths"] = {
        "raw": str(directory / "raw"),
        "metadata": str(directory / "metadata"),
        "processed": str(directory / "processed"),
        "reports": str(directory / "report"),
    }
    for section, values in overrides.items():
        if isinstance(values, dict):
            # A section the template does not carry is legal, so an optional one such
            # as the encoder can be declared by the test that needs it.
            payload[section] = {**payload.get(section, {}), **values}
        else:
            payload[section] = values

    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = load_config(write_config(tmp_path))
    cfg.paths.ensure()
    return cfg
