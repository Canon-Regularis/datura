"""Run the whole thing in one command.

Each stage delegates to the CLI that owns it, so this file cannot drift away from
what `python -m src.train.cnn` actually does. Stages are skipped when their output
is already on disk, which makes a rerun cheap and makes resuming after a failure
the default behaviour.

Usage:
    python -m src.pipeline --config configs/base.yaml
    python -m src.pipeline --config configs/base.yaml --skip-download
    python -m src.pipeline --config configs/base.yaml --only report
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.config import Config, load_config
from src.data import download as download_cli
from src.data import manifest as manifest_cli
from src.evaluate import explain as explain_cli
from src.evaluate import report as report_cli
from src.features import cache
from src.features import extract as extract_cli
from src.train import cnn as cnn_cli
from src.train import xgb as xgb_cli
from src.train.crossval import result_directory

# The CNN is trained at two capacities and the reported one is chosen on validation
# macro-F1. Explainability runs against the variant named here.
CNN_VARIANTS = (("cnn", "configs/cnn.yaml"), ("cnn_small", "configs/cnn_small.yaml"))
EXPLAINED_VARIANT = "cnn_small"
EXPLAINED_FOLD = "3"


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[], object]
    done: Callable[[], bool]


def build_stages(cfg: Config, config_path: str, *, skip_download: bool) -> list[Stage]:
    root = cfg.paths.raw / cfg.dataset.archive_root
    stages: list[Stage] = []

    download_args = ["--config", config_path]
    if skip_download:
        download_args.append("--skip-download")
    stages.append(
        Stage(
            "download",
            lambda: download_cli.main(download_args),
            lambda: root.exists() and all((root / name).exists() for name in cfg.dataset.species),
        )
    )
    stages.append(
        Stage(
            "manifest",
            lambda: manifest_cli.main(["--config", config_path]),
            lambda: manifest_cli.manifest_path(cfg).exists(),
        )
    )
    stages.append(
        Stage(
            "features",
            lambda: extract_cli.main(["--config", config_path]),
            lambda: cache.exists(cfg, "acoustic") and cache.exists(cfg, "logmel"),
        )
    )
    stages.append(
        Stage(
            "xgboost",
            lambda: xgb_cli.main(["--config", config_path]),
            lambda: (
                (result_directory(cfg, "xgboost") / "summary.csv").exists()
                and (result_directory(cfg, "metadata") / "summary.csv").exists()
            ),
        )
    )
    for name, model_config in CNN_VARIANTS:
        stages.append(
            Stage(
                name,
                lambda n=name, m=model_config: cnn_cli.main(
                    ["--config", config_path, "--model-config", m, "--name", n]
                ),
                lambda n=name: (result_directory(cfg, n) / "summary.csv").exists(),
            )
        )
    stages.append(
        Stage(
            "explain",
            lambda: explain_cli.main(
                [
                    "--config",
                    config_path,
                    "--model-config",
                    dict(CNN_VARIANTS)[EXPLAINED_VARIANT],
                    "--name",
                    EXPLAINED_VARIANT,
                    "--fold",
                    EXPLAINED_FOLD,
                ]
            ),
            lambda: (result_directory(cfg, EXPLAINED_VARIANT) / "occlusion.csv").exists(),
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


def run(stages: list[Stage], *, force: bool, only: str | None) -> int:
    selected = [s for s in stages if only is None or s.name == only]
    if only is not None and not selected:
        names = ", ".join(s.name for s in stages)
        raise SystemExit(f"unknown stage {only!r}; expected one of {names}")

    for stage in selected:
        if not force and stage.done():
            print(f"\n=== {stage.name}: already done, skipping ===")
            continue
        print(f"\n=== {stage.name} ===")
        started = time.perf_counter()
        code = stage.run()
        elapsed = time.perf_counter() - started
        if code:
            print(f"{stage.name} failed with exit code {code}")
            return int(code)
        print(f"--- {stage.name} finished in {elapsed / 60:.1f} min ---")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--skip-download", action="store_true", help="use an archive already on disk"
    )
    parser.add_argument("--force", action="store_true", help="rerun stages that are already done")
    parser.add_argument("--only", default=None, help="run a single stage by name")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure()
    stages = build_stages(cfg, args.config, skip_download=args.skip_download)

    print(f"pipeline for {cfg.name}: {' -> '.join(s.name for s in stages)}")
    started = time.perf_counter()
    code = run(stages, force=args.force, only=args.only)
    print(f"\ntotal {(time.perf_counter() - started) / 60:.1f} min")
    return code


if __name__ == "__main__":
    sys.exit(main())
