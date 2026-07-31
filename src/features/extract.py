"""Turn kept clips into cached feature arrays.

The same walk drives every representation. Adding one means registering it in
``build_extractor``; nothing else in this module changes.

Usage:
    python -m src.features.extract --config configs/base.yaml [--extractor acoustic|logmel|all]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio.io import load as load_audio
from src.audio.resample import to_target_rate
from src.audio.windows import split_into_windows
from src.config import Config, load_config
from src.data.manifest import load_manifest
from src.features import cache
from src.features.acoustic import AcousticFeatures
from src.features.base import FeatureExtractor
from src.features.spectrogram import LogMelSpectrogram

_INDEX_COLUMNS = [
    "clip_id",
    "tape_id",
    "species",
    "label",
    "year",
    "native_sample_rate",
    "duration_seconds",
    "bytes_on_disk",
    "window_index",
]


def build_extractor(kind: str, cfg: Config) -> FeatureExtractor:
    """Resolve a representation name to a configured extractor."""
    shared = {
        "n_fft": cfg.spectrogram.n_fft,
        "hop_length": cfg.spectrogram.hop_length,
        "n_mels": cfg.spectrogram.n_mels,
        "fmin": cfg.spectrogram.fmin,
        "fmax": cfg.spectrogram.fmax,
        "sample_rate": cfg.audio.target_sample_rate,
    }
    if kind == "acoustic":
        return AcousticFeatures(**shared)
    if kind == "logmel":
        return LogMelSpectrogram(**shared)
    raise ValueError(f"unknown extractor {kind!r}; expected 'acoustic' or 'logmel'")


def extract(cfg: Config, extractor: FeatureExtractor, manifest: pd.DataFrame) -> cache.FeatureStore:
    """Run one extractor over every kept clip and write the result to the cache."""
    root = cfg.paths.raw / cfg.dataset.archive_root
    shape = extractor.output_shape(cfg.audio.window_samples)
    writer = cache.FeatureWriter(cfg, extractor.name, shape, extractor.storage_dtype)

    index_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc=f"{extractor.name} [{cfg.name}]",
        unit="clip",
    ):
        try:
            signal, native_rate = load_audio(root / row.relative_path)
            resampled = to_target_rate(signal, native_rate, cfg.audio.target_sample_rate)
            windows = split_into_windows(
                resampled,
                cfg.audio.window_samples,
                cfg.audio.hop_samples,
                cfg.audio.pad_mode,
                cfg.audio.max_windows_per_clip,
            )
            block = extractor.transform_batch(windows, cfg.audio.target_sample_rate)
        except Exception as error:
            failures.append((row.clip_id, f"{type(error).__name__}: {error}"))
            continue

        writer.append(block)
        index_rows.extend(
            {
                "clip_id": row.clip_id,
                "tape_id": row.tape_id,
                "species": row.species,
                "label": int(row.label),
                "year": int(row.year),
                "native_sample_rate": int(row.native_sample_rate),
                "duration_seconds": float(row.duration_seconds),
                "bytes_on_disk": int(row.bytes_on_disk),
                "window_index": position,
            }
            for position in range(len(block))
        )

    store = writer.close(pd.DataFrame(index_rows, columns=_INDEX_COLUMNS))

    if failures:
        print(f"\n{len(failures)} clip(s) could not be processed:")
        for clip_id, reason in failures[:10]:
            print(f"  {clip_id}: {reason}")
    return store


def _summarise(store: cache.FeatureStore, extractor: FeatureExtractor) -> None:
    per_species = (
        store.index.groupby("species")
        .agg(
            windows=("clip_id", "size"),
            clips=("clip_id", "nunique"),
            tapes=("tape_id", "nunique"),
        )
        .reset_index()
    )
    print(f"\n{extractor.name}: {store.features.shape} {store.features.dtype}")
    print(per_species.to_string(index=False))
    size_mb = np.prod(store.features.shape) * store.features.dtype.itemsize / 1e6
    print(f"cache size {size_mb:.0f} MB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--extractor", default="all", choices=["acoustic", "logmel", "all"])
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure()
    manifest = load_manifest(cfg, kept_only=True)
    print(f"{len(manifest)} kept clips across {manifest['tape_id'].nunique()} tapes")

    kinds = ["acoustic", "logmel"] if args.extractor == "all" else [args.extractor]
    for kind in kinds:
        if cache.exists(cfg, kind) and not args.force:
            print(f"{kind}: cache present, skipping (use --force to rebuild)")
            continue
        extractor = build_extractor(kind, cfg)
        store = extract(cfg, extractor, manifest)
        _summarise(store, extractor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
