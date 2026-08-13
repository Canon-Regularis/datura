"""Turn kept clips into cached feature arrays.

The same walk drives every representation. Adding one means registering it in
``src.features.registry``; nothing else in this module changes.

Usage:
    python -m src.features.extract --config configs/base.yaml [--extractor acoustic|logmel|all]
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import cli
from src.audio.decode import decode_to_windows
from src.config import Config
from src.data.manifest import load_manifest
from src.errors import DaturaError
from src.features import cache, registry
from src.features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class ExtractionIncomplete(DaturaError):
    """Raised when clips were dropped while building a cache.

    A cache missing clips is not obviously wrong from the outside. It trains, it
    scores, and it quietly compares two representations on different corpora if only
    one of them dropped a file. The probe and the networks share folds because their
    two caches cover the same clips, and that only holds while this raises.
    """


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


def extract(
    cfg: Config,
    extractor: FeatureExtractor,
    manifest: pd.DataFrame,
    allow_failures: bool = False,
) -> cache.FeatureStore:
    """Run one extractor over every kept clip and write the result to the cache."""
    root = cfg.paths.raw / cfg.dataset.archive_root
    shape = extractor.output_shape(cfg.audio.window_samples)
    writer = cache.FeatureWriter(cfg, extractor, shape, extractor.storage_dtype)

    index_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc=f"{extractor.name} [{cfg.name}]",
        unit="clip",
    ):
        try:
            windows = decode_to_windows(cfg, root / row.relative_path)
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
        record = cfg.paths.metadata / f"extract_failures_{cfg.corpus}_{extractor.name}.csv"
        pd.DataFrame(failures, columns=["clip_id", "reason"]).to_csv(record, index=False)
        logger.warning("\n%d clip(s) could not be processed, listed in %s:", len(failures), record)
        for clip_id, reason in failures[:10]:
            logger.warning("  %s: %s", clip_id, reason)
        if not allow_failures:
            raise ExtractionIncomplete(
                f"{len(failures)} of {len(manifest)} clips failed for {extractor.name}; "
                f"see {record.name}, or pass --allow-failures to build the cache without them"
            )
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
    logger.info("\n%s: %s %s", extractor.name, store.features.shape, store.features.dtype)
    logger.info(per_species.to_string(index=False))
    size_mb = np.prod(store.features.shape) * store.features.dtype.itemsize / 1e6
    logger.info("cache size %.0f MB", size_mb)


def extract_all(
    cfg: Config,
    *,
    kinds: list[str] | None = None,
    force: bool = False,
    allow_failures: bool = False,
) -> None:
    """Fill the cache for every representation this configuration needs.

    ``kinds`` defaults to all of them. A representation whose cache is already present
    is skipped, which is what makes a rerun after a failure cheap.
    """
    manifest = load_manifest(cfg, kept_only=True)
    logger.info("%d kept clips across %d tapes", len(manifest), manifest["tape_id"].nunique())

    for kind in kinds if kinds is not None else list(registry.kinds()):
        extractor = registry.build_extractor(kind, cfg)
        if cache.exists(cfg, extractor) and not force:
            logger.info("%s: cache present, skipping (use --force to rebuild)", kind)
            continue
        _summarise(extract(cfg, extractor, manifest, allow_failures), extractor)


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    parser.add_argument(
        "--extractor",
        default="all",
        choices=["all", *registry.kinds()],
        help="which representation to build",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="write the cache even if some clips could not be read, rather than refusing",
    )
    args = parser.parse_args(argv)

    extract_all(
        cli.prepare(args),
        kinds=None if args.extractor == "all" else [args.extractor],
        force=args.force,
        allow_failures=args.allow_failures,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
