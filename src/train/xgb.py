"""Train the gradient boosted baselines: acoustic descriptors, then the control.

Both run here because they use the same estimator. Any gap between them comes from
the features rather than from the learner, which is exactly the comparison the
control exists to make.

Usage:
    python -m src.train.xgb [--config configs/base.yaml]
"""

from __future__ import annotations

import sys

from src import cli
from src.features import registry as features
from src.models import registry as models
from src.models.registry import load_settings
from src.train.session import assemble, train


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    parser.add_argument("--skip-control", action="store_true", help="omit the metadata control")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="rerun the whole split under fresh seeds, for more estimates of the same quantity",
    )
    args = parser.parse_args(argv)

    cfg = cli.prepare(args)
    assembly = assemble(cfg, features.ACOUSTIC, repeats=args.repeats)

    wanted = [models.get("xgboost")]
    if not args.skip_control:
        wanted.append(models.control())

    for spec in wanted:
        train(cfg, spec, assembly, load_settings(spec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
