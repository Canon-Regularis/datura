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
from functools import partial

from src import cli
from src.config import Config
from src.data import annotations as annotations_cli
from src.data.download import fetch
from src.data.manifest import write_manifest
from src.errors import DaturaError
from src.evaluate.diagnostics import build as build_diagnostics
from src.evaluate.explain import explain_result
from src.evaluate.report import build as build_report
from src.features import registry as features
from src.features.extract import extract_all
from src.models import registry as models
from src.results import (
    config_directory,
    diagnostics_path,
    has_results,
    manifest_path,
    occlusion_path,
)
from src.train.calltypes import run_call_types
from src.train.cnn import train_network
from src.train.xgb import train_trees

logger = logging.getLogger(__name__)

# Which trained variant the explainability stage runs against, and on which fold.
EXPLAINED_VARIANT = "cnn_small"
EXPLAINED_FOLD = 3

# How many times a call type task redraws its split. The tree models carry this in
# the registry, beside the cost that decides it; a call type task is the same trees
# on a subset, so it takes the same count from the model it fits.
CALL_TYPE_REPEATS = models.get("xgboost").repeats

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


def _acquisition_stages(cfg: Config, *, skip_download: bool) -> list[Stage]:
    root = cfg.paths.raw / cfg.dataset.archive_root
    return [
        Stage(
            "download",
            lambda: fetch(cfg, skip_download=skip_download),
            lambda: root.exists() and all((root / name).exists() for name in cfg.dataset.species),
        ),
        Stage(
            # Before the manifest, because the audit tables describing the field
            # notes are built as part of it.
            "annotations",
            lambda: annotations_cli.build(cfg),
            lambda: annotations_cli.annotations_path(cfg).exists(),
        ),
        Stage(
            "manifest",
            lambda: write_manifest(cfg),
            lambda: manifest_path(cfg).exists(),
        ),
        Stage(
            "features",
            lambda: extract_all(cfg),
            lambda: all(features.cache_exists(kind, cfg) for kind in features.kinds()),
        ),
    ]


def _training_stages(cfg: Config) -> list[Stage]:
    """One stage per model, taken straight from the registry.

    The tree models share a single command, because the control has to be fitted on
    the same folds as the baseline it is compared against.
    """
    wanted = [spec for spec in models.trained_by(models.TREES) if cfg.pipeline.allows(spec.name)]
    if not wanted:
        return []

    trees = [spec.name for spec in wanted]
    stages = [
        Stage(
            models.TREES,
            partial(train_trees, cfg),
            lambda: all(has_results(cfg, name) for name in trees),
        )
    ]
    for spec in models.trained_by(models.NETWORK):
        if not cfg.pipeline.allows(spec.name):
            continue
        stages.append(
            Stage(
                spec.name,
                partial(train_network, cfg, spec.name, repeats=spec.repeats),
                partial(has_results, cfg, spec.name),
            )
        )
    return stages


def _call_type_stages(cfg: Config) -> list[Stage]:
    """The within species call type tasks, one stage per species.

    These were run by hand for a long time, which meant a full pipeline run
    reproduced the species results and none of the call type ones, while the report
    printed both.
    """
    return [
        Stage(
            f"calltypes_{species.lower()}",
            partial(run_call_types, cfg, species, repeats=CALL_TYPE_REPEATS),
            partial(_has_call_type_results, cfg, species),
        )
        for species in cfg.pipeline.call_type_species(CALL_TYPE_SPECIES)
    ]


def _has_call_type_results(cfg: Config, species: str) -> bool:
    """Whether this species has been posed its call type questions already."""
    directory = config_directory(cfg)
    existing = [child.name for child in directory.iterdir()] if directory.exists() else []
    return any(
        name.startswith(f"calltype_{species.lower()}_") and has_results(cfg, name)
        for name in existing
    )


def _reporting_stages(cfg: Config) -> list[Stage]:
    stages = []

    # Explaining a model this configuration never trains would ask for a checkpoint
    # that cannot exist, so the stage is only built where the model is.
    if cfg.pipeline.allows(EXPLAINED_VARIANT):
        stages.append(
            Stage(
                "explain",
                lambda: explain_result(cfg, EXPLAINED_VARIANT, fold_index=EXPLAINED_FOLD),
                lambda: occlusion_path(cfg, EXPLAINED_VARIANT).exists(),
            )
        )

    # Before the report, because the report's own tables do not answer this and the
    # README quotes it beside them. Skipped when it is already on disk: it fits about a
    # hundred and twenty models and nothing upstream of it changes what they see.
    stages.append(
        Stage(
            "diagnostics",
            lambda: build_diagnostics(cfg),
            lambda: diagnostics_path(cfg).exists(),
        )
    )

    stages.append(
        Stage(
            "report",
            lambda: build_report(cfg),
            # The report is cheap and summarises everything before it, so it always reruns.
            lambda: False,
        )
    )
    return stages


def build_stages(cfg: Config, *, skip_download: bool) -> list[Stage]:
    """Every stage this configuration runs, in the order it runs them.

    Each stage closes over the loaded configuration and calls the work directly. It
    went through the command line once, which reparsed the same YAML twelve times a
    run and made a stage's arguments unreachable to a type checker.
    """
    return [
        *_acquisition_stages(cfg, skip_download=skip_download),
        *_training_stages(cfg),
        *_call_type_stages(cfg),
        *_reporting_stages(cfg),
    ]


def run(stages: list[Stage], *, force: bool, only: str | None) -> None:
    """Run the selected stages in order, stopping at the first one that raises.

    There is no exit code to inspect any more. Every stage used to be a ``main(argv)``
    returning one, and a stage that failed had to remember to return non-zero for the
    run to stop; now each calls a function that either finishes or raises, and the
    error carries what went wrong rather than the number 1.
    """
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
        stage.run()
        elapsed = time.perf_counter() - started
        logger.info("--- %s finished in %.1f min ---", stage.name, elapsed / 60)


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    parser.add_argument(
        "--skip-download", action="store_true", help="use an archive already on disk"
    )
    parser.add_argument("--force", action="store_true", help="rerun stages that are already done")
    parser.add_argument("--only", default=None, help="run a single stage by name")
    args = parser.parse_args(argv)

    cfg = cli.prepare(args)
    stages = build_stages(cfg, skip_download=args.skip_download)
    logger.info("pipeline for %s: %s", cfg.name, " -> ".join(stage.name for stage in stages))

    started = time.perf_counter()
    run(stages, force=args.force, only=args.only)
    logger.info("\ntotal %.1f min", (time.perf_counter() - started) / 60)
    return 0


if __name__ == "__main__":  # pragma: no cover
    # A stage that fails raises, and the run stops there with a non-zero exit. It used
    # to return an exit code through argparse and this printed the number 1; the error
    # says which stage and why, which is what somebody three hours into a run needs.
    try:
        sys.exit(main())
    except DaturaError as error:
        logger.error("%s", error)
        sys.exit(1)
