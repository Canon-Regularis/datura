"""The command line around the prediction path.

Usage:
    python -m src.predict recording.wav [--model xgboost] [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.errors import DaturaError
from src.evaluate import coverage
from src.models import registry as models
from src.predict import __doc__ as description
from src.predict.inference import CannotPredict, probabilities
from src.predict.policy import DEFAULT_FOLD, DEFAULT_MODEL, TARGET_ACCURACY, standing
from src.predict.report import render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(description or "").splitlines()[0])
    parser.add_argument("recording", type=Path, help="a wav file to identify")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "one model, or several comma separated to average them. "
            f"'{coverage.ENSEMBLE_NAME}' is the best measured combination and needs the "
            "encoder weights; the default runs offline"
        ),
    )
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=TARGET_ACCURACY,
        help="decline any prediction not confident enough to have earned this accuracy",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if not args.recording.exists():
        raise CannotPredict(f"{args.recording} not found")

    chosen = [name.strip() for name in args.model.split(",") if name.strip()]
    for name in chosen:
        models.get(name)  # raises UnknownModel with the roster, before any audio is read

    # A committed curve exists for the pair under the name it is stored by, so an
    # ensemble is measured rather than guessed at, exactly like a single model.
    label = "+".join(chosen)
    scores = probabilities(cfg, args.recording, chosen, args.fold)
    verdict = standing(cfg, label, float(scores.max()), args.target_accuracy)
    print(render(cfg, label, scores, verdict, args.fold))
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    try:
        sys.exit(main())
    except DaturaError as error:
        print(f"\n  {error}\n", file=sys.stderr)
        sys.exit(1)
