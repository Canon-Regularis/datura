"""Run the whole thing in one command.

Each stage delegates to the entry point that owns it, so this file cannot drift
away from what those commands actually do. Stages are skipped when their output is
already on disk, which makes a rerun cheap and makes resuming after a failure the
default behaviour.

The stage list is derived from the model registry: adding a model adds a stage.

Usage:
    python -m src.pipeline --config configs/base.yaml
    python -m src.pipeline --config configs/base.yaml --skip-download
    python -m src.pipeline --config configs/base.yaml --only report
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from src import cli
from src.config import Config
from src.data import annotations as annotations_cli
from src.data import download as download_cli
from src.data import manifest as manifest_cli
from src.errors import DaturaError
from src.evaluate import explain as explain_cli
from src.evaluate import report as report_cli
from src.features import extract as extract_cli
from src.features import registry as features
from src.models import registry as models
from src.results import config_directory, has_results, model_directory
from src.train import calltypes as calltypes_cli
from src.train import cnn as cnn_cli
from src.train import xgb as xgb_cli

logger = logging.getLogger(__name__)

# Which trained variant the explainability stage runs against, and on which fold.
EXPLAINED_VARIANT = "cnn_small"
EXPLAINED_FOLD = "3"

# How many times a call type task redraws its split. The tree models carry this in
# the registry, beside the cost that decides it; a call type task is the same trees
# on a subset, so it takes the same count from the model it fits.
CALL_TYPE_REPEATS = str(models.get("xgboost").repeats)

# Which species the call type tasks are posed inside. Every other species in the
# study has no call type reaching the minimum of 60 clips over 10 tapes.
CALL_TYPE_SPECIES = ("SpermWhale", "KillerWhale")


class UnknownStage(DaturaError):
    """Raised when --only names a stage that does not exist."""


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline: what to run, and how to tell it already ran."""

    name: str
    run: Callable[[], object]
    done: Callable[[], bool]


def _acquisition_stages(cfg: Config, config_path: str, skip_download: bool) -> list[Stage]:
    root = cfg.paths.raw / cfg.dataset.archive_root
    download_args = ["--config", config_path]
    if skip_download:
        download_args.append("--skip-download")

    return [
        Stage(
            "download",
            lambda: download_cli.main(download_args),
            lambda: root.exists() and all((root / name).exists() for name in cfg.dataset.species),
        ),
        Stage(
            # Before the manifest, because the audit tables describing the field
            # notes are built as part of it.
            "annotations",
            lambda: annotations_cli.main(["--config", config_path]),
            lambda: annotations_cli.annotations_path(cfg).exists(),
        ),
        Stage(
            "manifest",
            lambda: manifest_cli.main(["--config", config_path]),
            lambda: manifest_cli.manifest_path(cfg).exists(),
        ),
        Stage(
            "features",
            lambda: extract_cli.main(["--config", config_path]),
            lambda: all(features.cache_exists(kind, cfg) for kind in features.kinds()),
        ),
    ]


def _training_stages(cfg: Config, config_path: str) -> list[Stage]:
    """One stage per model, taken straight from the registry.

    The tree models share a single command, because the control has to be fitted on
    the same folds as the baseline it is compared against.
    """
    wanted = [spec for spec in models.trained_by(models.TREES) if cfg.pipeline.allows(spec.name)]
    if not wanted:
        return []

    # One command fits every tree on one assembly, so they have to agree on how many
    # times the split is redrawn. Disagreeing would mean the control and the model it
    # is compared against were scored on different numbers of splits.
    counts = {spec.repeats for spec in wanted}
    if len(counts) != 1:
        raise ValueError(f"the tree models declare different repeat counts: {sorted(counts)}")
    repeats = str(next(iter(counts)))

    trees = [spec.name for spec in wanted]
    stages = [
        Stage(
            models.TREES,
            lambda n=repeats: xgb_cli.main(["--config", config_path, "--repeats", n]),
            lambda: all(has_results(cfg, name) for name in trees),
        )
    ]
    for spec in models.trained_by(models.NETWORK):
        if not cfg.pipeline.allows(spec.name):
            continue
        stages.append(
            Stage(
                spec.name,
                lambda n=spec.name, r=str(spec.repeats): cnn_cli.main(
                    ["--config", config_path, "--name", n, "--repeats", r]
                ),
                lambda n=spec.name: has_results(cfg, n),
            )
        )
    return stages


def _call_type_stages(cfg: Config, config_path: str) -> list[Stage]:
    """The within species call type tasks, one stage per species.

    These were run by hand for a long time, which meant a full pipeline run
    reproduced the species results and none of the call type ones, while the report
    printed both.
    """
    return [
        Stage(
            f"calltypes_{species.lower()}",
            lambda s=species: calltypes_cli.main(
                ["--config", config_path, "--species", s, "--repeats", CALL_TYPE_REPEATS]
            ),
            lambda s=species: any(
                name.startswith(f"calltype_{s.lower()}_") and has_results(cfg, name)
                for name in _existing_results(cfg)
            ),
        )
        for species in cfg.pipeline.call_type_species(CALL_TYPE_SPECIES)
    ]


def _existing_results(cfg: Config) -> list[str]:
    directory = config_directory(cfg)
    return [child.name for child in directory.iterdir()] if directory.exists() else []


def _reporting_stages(cfg: Config, config_path: str) -> list[Stage]:
    stages = []

    # Explaining a model this configuration never trains would ask for a checkpoint
    # that cannot exist, so the stage is only built where the model is.
    if cfg.pipeline.allows(EXPLAINED_VARIANT):
        stages.append(
            Stage(
                "explain",
                lambda: explain_cli.main(
                    [
                        "--config",
                        config_path,
                        "--name",
                        EXPLAINED_VARIANT,
                        "--fold",
                        EXPLAINED_FOLD,
                    ]
                ),
                lambda: (model_directory(cfg, EXPLAINED_VARIANT) / "occlusion.csv").exists(),
            )
        )

    stages.append(
        Stage(
            "report",
            lambda: report_cli.main(["--config", config_path]),
            # The report is cheap and summarises everything before it, so it always reruns.
            lambda: False,
        )
    )
    return stages


def build_stages(cfg: Config, config_path: str, *, skip_download: bool) -> list[Stage]:
    return [
        *_acquisition_stages(cfg, config_path, skip_download),
        *_training_stages(cfg, config_path),
        *_call_type_stages(cfg, config_path),
        *_reporting_stages(cfg, config_path),
    ]


def run(stages: list[Stage], *, force: bool, only: str | None) -> int:
    selected = [stage for stage in stages if only is None or stage.name == only]
    if only is not None and not selected:
        names = ", ".join(stage.name for stage in stages)
        raise UnknownStage(f"unknown stage {only!r}; expected one of {names}")

    for stage in selected:
        if not force and stage.done():
            logger.info("\n=== %s: already done, skipping ===", stage.name)
            continue

        logger.info("\n=== %s ===", stage.name)
        started = time.perf_counter()
        code = stage.run()
        elapsed = time.perf_counter() - started
        if code:
            logger.error("%s failed with exit code %s", stage.name, code)
            return int(code)
        logger.info("--- %s finished in %.1f min ---", stage.name, elapsed / 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    parser.add_argument(
        "--skip-download", action="store_true", help="use an archive already on disk"
    )
    parser.add_argument("--force", action="store_true", help="rerun stages that are already done")
    parser.add_argument("--only", default=None, help="run a single stage by name")
    args = parser.parse_args(argv)

    cfg = cli.prepare(args)
    stages = build_stages(cfg, args.config, skip_download=args.skip_download)
    logger.info("pipeline for %s: %s", cfg.name, " -> ".join(stage.name for stage in stages))

    started = time.perf_counter()
    code = run(stages, force=args.force, only=args.only)
    logger.info("\ntotal %.1f min", (time.perf_counter() - started) / 60)
    return code


if __name__ == "__main__":
    sys.exit(main())
