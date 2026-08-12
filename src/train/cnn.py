"""Train the log mel network, on the same folds as the baselines.

Usage:
    python -m src.train.cnn [--config configs/base.yaml] [--name cnn_small]
"""

from __future__ import annotations

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
) -> Path:
    """Fit one network trainer model on one configuration.

    ``repeats`` defaults to what the registry declares, because the cost that decides
    it belongs beside the model rather than in whoever calls this.
    """
    spec = models.get(name)
    overrides = {"train": {"epochs": epochs}} if epochs is not None else None
    settings = load_settings(spec, overrides)
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
    args = parser.parse_args(argv)

    train_network(cli.prepare(args), args.name, epochs=args.epochs, repeats=args.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
