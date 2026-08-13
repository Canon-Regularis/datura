"""Train the log mel network, on the same folds as the baselines.

Usage:
    python -m src.train.cnn [--config configs/base.yaml] [--name cnn_small]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from src import cli
from src.config import Config
from src.models import registry as models
from src.models.cnn import resolve_device
from src.models.registry import load_settings
from src.train.session import Assembly, assemble, train

logger = logging.getLogger(__name__)


def log_device(settings: dict) -> torch.device:
    """Say what the run is training on. Results shift slightly between the two."""
    device = resolve_device(str(settings.get("device", "auto")))
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        logger.info("training on %s, %.1f GB", properties.name, properties.total_memory / 1e9)
    else:
        logger.info("training on CPU with %d threads", torch.get_num_threads())
    return device


def train_network(
    cfg: Config,
    name: str,
    *,
    epochs: int | None = None,
    repeats: int | None = None,
    deterministic: bool | None = None,
) -> Path:
    """Fit one network trainer model on one configuration.

    ``repeats`` defaults to what the registry declares, because the cost that decides
    it belongs beside the model rather than in whoever calls this.

    ``deterministic`` overrides the config, and exists so that a change to the training
    loop can be checked. These models are not reproducible as shipped: cuDNN picks
    whichever kernel is quickest on the day, so refitting a fold disagrees with the
    committed predictions on roughly one clip in six. Forcing it on costs about a third
    of the throughput and is not how the published numbers were produced, which is why
    it is a flag rather than a setting change.
    """
    spec = models.get(name)
    overrides: dict[str, dict[str, object]] = {"train": {}}
    if epochs is not None:
        overrides["train"]["epochs"] = epochs
    if deterministic is not None:
        overrides["train"]["deterministic"] = deterministic
    settings = load_settings(spec, overrides if overrides["train"] else None)
    log_device(settings["train"])

    assembly: Assembly = assemble(
        cfg, spec.source, repeats=repeats if repeats is not None else spec.repeats
    )
    return train(cfg, spec, assembly, settings, name=name)


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    cli.add_variant_name(parser)
    parser.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="rerun the whole split under fresh seeds; defaults to what the registry declares",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "force identical reruns on the same hardware, at about a third less "
            "throughput. Off in the shipped configs, so the published numbers were not "
            "produced with it; use it to check that a change left the model alone"
        ),
    )
    args = parser.parse_args(argv)

    train_network(
        cli.prepare(args),
        args.name,
        epochs=args.epochs,
        repeats=args.repeats,
        deterministic=args.deterministic,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
