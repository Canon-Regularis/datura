"""Train every gradient boosted model: acoustic descriptors, then the no-audio ones.

They all run here because they use the same estimator on the same folds. Any gap
between them comes from the features rather than from the learner, which is exactly
the comparison the controls exist to make.

``--only`` trains one of them and leaves the rest alone. Results already on disk are
compared against numbers already published, and refitting a model to get at a
different one would move both.

Usage:
    python -m src.train.xgb [--config configs/base.yaml] [--only logbook] [--repeats 10]
"""

from __future__ import annotations

import sys
from pathlib import Path

from src import cli
from src.config import Config
from src.features import registry as features
from src.models import registry as models
from src.models.registry import ModelSpec, load_settings
from src.train.session import assemble, train


def train_trees(
    cfg: Config,
    *,
    only: str | None = None,
    skip_control: bool = False,
    repeats: int = 1,
) -> list[Path]:
    """Fit every tree model on one configuration, and say where each landed.

    A typed function rather than only a command, because the pipeline used to reach
    this logic by building an argv list and handing it to ``main``. That put argparse
    between two pieces of the same program: a renamed flag failed at runtime instead of
    at import, integers were stringified so they could be parsed back, and the logic
    below could not be called or tested except through a command line.
    """
    wanted = list(models.trained_by(models.TREES))
    if only:
        wanted = [models.get(only)]
    elif skip_control:
        wanted = [spec for spec in wanted if not spec.is_control]

    # Grouped by the cached representation each model reads, rather than assuming every
    # tree reads the acoustic descriptors. A model whose source is not cached audio
    # takes its columns off the window index, which every assembly carries, so it can
    # ride along with any of them.
    by_source: dict[str, list[ModelSpec]] = {}
    for spec in wanted:
        source = spec.source if spec.hears_audio else features.ACOUSTIC
        by_source.setdefault(source, []).append(spec)

    written = []
    for source, specs in sorted(by_source.items()):
        assembly = assemble(cfg, source, repeats=repeats)
        written += [train(cfg, spec, assembly, load_settings(spec)) for spec in specs]
    return written


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    parser.add_argument("--skip-control", action="store_true", help="omit the metadata control")
    parser.add_argument(
        "--only",
        default=None,
        help="train one of these models rather than all of them, leaving the others on disk",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="rerun the whole split under fresh seeds, for more estimates of the same quantity",
    )
    args = parser.parse_args(argv)

    train_trees(
        cli.prepare(args),
        only=args.only,
        skip_control=args.skip_control,
        repeats=args.repeats,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
