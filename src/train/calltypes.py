"""Call type classification, within one species at a time.

Several call types sit almost entirely inside one species: coda appears only in
sperm whale, tonal and squeak only in killer whale, song only in humpback. A model
trained across species would be relearning species under another name, which is the
mistake the first phase already measured. So every task here is posed inside a
single species, and the only question asked is which call a clip contains.

Call types overlap, so the task is multi label rather than multi class: 145 sperm
whale clips are both click and coda. Each type therefore gets its own binary model,
which keeps every type comparable to every other and reuses the runner unchanged.

Each task is reported against a control that sees everything written down about the
recording and no audio at all. Location alone identifies the species for 98% of clips
in this collection and the collection code does better still, so a call type result
that does not clear the same bar is not evidence about the animal.

The control was narrower once, and it cost this project its only positive result. It
saw the site, the coordinates, the collection and the noise conditions, but none of
the four header fields, so it was denied clip duration. A note is written against a
whole cut, so duration predicts most call type labels before anything is heard, and a
control without it is clearing a lower bar than the audio model beside it.

Usage:
    python -m src.train.calltypes [--config configs/base.yaml] [--species SpermWhale]
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from src import cli
from src.config import Config
from src.data import annotations as ann
from src.data.notes import load_vocabulary
from src.data.splits import folds_for_index
from src.features import registry as features
from src.features.controls import LogbookFeatureSource
from src.features.source import DerivedSource, FeatureSource
from src.models import registry as models
from src.models.registry import load_settings
from src.train.crossval import run_cross_validation, save_result
from src.train.folds import FoldPlan
from src.train.tasks import (
    ABSENT,
    CLASS_NAMES,
    PRESENT,
    CallTypeError,
    Task,
    clip_labels,
    viable_tasks,
    window_index,
)

logger = logging.getLogger(__name__)

# Re-exported so the entry point stays the one name a reader looks for, while the
# question of which tasks are viable can be asked without the training stack.
__all__ = [
    "ABSENT",
    "CLASS_NAMES",
    "PRESENT",
    "CallTypeError",
    "Task",
    "clip_labels",
    "run_task",
    "viable_tasks",
    "window_index",
]


def run_task(
    cfg: Config,
    base: FeatureSource,
    labels: pd.DataFrame,
    task: Task,
    model_name: str = "xgboost",
    suffix: str = "",
    repeats: int = 1,
    skip_control: bool = False,
) -> None:
    """Fit the audio model and its context control on identical folds.

    The control is always trees. It is a floor rather than a contender, and holding
    it fixed keeps the margin comparable across whichever model is on trial.

    Both sides run the same plan, so repeat three fold two of the audio model is the
    same split as repeat three fold two of the control. That pairing is what the
    comparison rests on.

    ``skip_control`` matters more than it looks. One control serves every model on a
    task, so fitting a second model would otherwise refit it, and a run with fewer
    repeats than the first would quietly replace fifty splits with five. Every model
    on that task then gets compared on the five they share.
    """
    subset, positions = window_index(base, labels, task)
    folds = folds_for_index(subset, cfg)
    plan = FoldPlan.repeated(cfg, subset, repeats) if repeats > 1 else FoldPlan.single(folds)

    audio = DerivedSource(base, subset, positions, name=f"{base.name}_{task.call_type}")
    context_index = ann.attach_context(subset, labels)
    # Everything written down about the recording, which is the same control the
    # species task is measured against. A narrower one used to be used here and it
    # was denied clip duration, the single field that most predicts whether a cut
    # contains a given call.
    control = LogbookFeatureSource(context_index, ann.condition_columns(context_index))

    positive_clips = subset[subset["label"] == 1]["clip_id"].nunique()
    logger.info(
        "\n%s: %d clips carry it across %d tapes, %d windows in total",
        task.name,
        positive_clips,
        task.tapes,
        len(subset),
    )

    audio_spec = models.get(model_name)
    tag = suffix + ("" if model_name == "xgboost" else f"_{model_name}")

    wanted = [(audio_spec, audio, f"{task.name}{tag}")]
    if not skip_control:
        wanted.append((models.control(), control, f"{task.control_name}{suffix}"))

    for spec, source, name in wanted:
        settings = load_settings(spec)
        result = run_cross_validation(
            cfg,
            source,
            plan,
            lambda sp=spec, st=settings: sp.build(cfg, st),
            name,
            class_names=list(CLASS_NAMES),
        )
        save_result(cfg, result, extra={"max_clip_seconds": task.max_clip_seconds})
        logger.info("%s", result.headline())


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    parser.add_argument(
        "--species",
        default="SpermWhale",
        help="which species to pose the call type questions inside",
    )
    parser.add_argument(
        "--model", default="xgboost", help="which registered model to fit, such as cnn_small"
    )
    parser.add_argument(
        "--only", default=None, help="run one call type rather than every viable one"
    )
    parser.add_argument(
        "--max-clip-seconds",
        type=float,
        default=None,
        help="drop clips longer than this, where a whole cut label stops describing a window",
    )
    parser.add_argument(
        "--suffix", default="", help="tag appended to result names, to keep runs side by side"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="how many times to rerun the whole split under a shifted seed",
    )
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="leave the context control alone, for fitting a second model on a task",
    )
    args = parser.parse_args(argv)

    cfg = cli.prepare(args)
    vocabulary = load_vocabulary()
    labels = clip_labels(cfg, args.species, args.max_clip_seconds)
    tasks = viable_tasks(cfg, args.species, labels, vocabulary)
    logger.info(
        "%s: %d viable call types\n%s",
        args.species,
        len(tasks),
        "\n".join(f"  {task!r}" for task in tasks),
    )

    spec = models.get(args.model)
    base = features.load_source(spec.source, cfg)
    for task in tasks:
        if args.only and task.call_type != args.only:
            continue
        run_task(
            cfg,
            base,
            labels,
            task,
            model_name=args.model,
            suffix=args.suffix,
            repeats=args.repeats,
            skip_control=args.skip_control,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
