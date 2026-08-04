"""Call type classification, within one species at a time.

Several call types sit almost entirely inside one species: coda appears only in
sperm whale, tonal and squeak only in killer whale, song only in humpback. A model
trained across species would be relearning species under another name, which is the
mistake the first phase already measured. So every task here is posed inside a
single species, and the only question asked is which call a clip contains.

Call types overlap, so the task is multi label rather than multi class: 145 sperm
whale clips are both click and coda. Each type therefore gets its own binary model,
which keeps every type comparable to every other and reuses the runner unchanged.

Each task is reported against a context control that sees the site, the coordinates,
the collection the cut came from and the noise conditions, and no audio at all.
Location alone identifies the species for 98% of clips in this collection and the
collection code does better still, so a call type result that does not clear the
same bar is not evidence about the animal.

Usage:
    python -m src.train.calltypes [--config configs/base.yaml] [--species SpermWhale]
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from src import cli
from src.config import Config
from src.data import annotations as ann
from src.data.manifest import load_manifest
from src.data.notes import CALL_PREFIX, Vocabulary, load_vocabulary
from src.data.splits import folds_for_index
from src.errors import DaturaError
from src.features import registry as features
from src.features.source import ContextFeatureSource, DerivedSource, FeatureSource
from src.models import registry as models
from src.models.registry import load_settings
from src.train.crossval import run_cross_validation, save_result
from src.train.folds import FoldPlan

logger = logging.getLogger(__name__)

ABSENT = "absent"
PRESENT = "present"
CLASS_NAMES = (ABSENT, PRESENT)

# A call type needs enough independent recordings on both sides to be worth fitting.
# Humpback song sits on two tapes and social sound on one, which would measure
# memorisation of a single recording rather than anything about the call.
MINIMUM_TAPES = 10
MINIMUM_CLIPS = 60


class CallTypeError(DaturaError):
    """Raised when no call type in a species has enough independent recordings."""


class Task:
    """One binary question: does this clip contain this call type?"""

    def __init__(
        self,
        species: str,
        call_type: str,
        positives: int,
        tapes: int,
        max_clip_seconds: float | None = None,
    ):
        self.species = species
        self.call_type = call_type
        self.positives = positives
        self.tapes = tapes
        self.max_clip_seconds = max_clip_seconds

    @property
    def name(self) -> str:
        return f"calltype_{self.species.lower()}_{self.call_type}"

    @property
    def control_name(self) -> str:
        return f"{self.name}_context"

    def __repr__(self) -> str:
        guard = "" if self.max_clip_seconds is None else f", clips under {self.max_clip_seconds:g}s"
        return (
            f"Task({self.species}, {self.call_type}, "
            f"{self.positives} clips, {self.tapes} tapes{guard})"
        )


def clip_labels(cfg: Config, species: str, max_clip_seconds: float | None = None) -> pd.DataFrame:
    """One row per kept clip of this species, carrying every call type flag.

    ``max_clip_seconds`` drops the long recordings, and it exists because the labels
    are written against a whole cut rather than against a moment in it. A coda
    labelled clip runs to a median of 64 seconds and a maximum of 24 minutes, while
    a coda itself lasts a few seconds. Windows sampled across such a recording
    inherit a label that most of them do not deserve, so a short clip carries a far
    more honest label than a long one.
    """
    manifest = load_manifest(cfg, kept_only=True)
    parsed = ann.load(cfg)
    columns = ann.call_columns(parsed)
    context = [
        "site",
        "latitude",
        "longitude",
        "collection_code",
        *ann.condition_columns(parsed),
    ]

    joined = manifest.merge(parsed[["clip_id", *columns, *context]], on="clip_id", how="left")
    subset = joined[joined["species"] == species]
    if max_clip_seconds is not None:
        before = len(subset)
        subset = subset[subset["duration_seconds"] <= max_clip_seconds]
        logger.info(
            "clips of %s seconds or less: %d of %d kept",
            max_clip_seconds,
            len(subset),
            before,
        )

    subset = subset.reset_index(drop=True)
    if subset.empty:
        raise CallTypeError(f"no kept clips for {species}")
    return subset


def viable_tasks(
    cfg: Config, species: str, labels: pd.DataFrame, vocabulary: Vocabulary | None = None
) -> list[Task]:
    """Call types with enough clips and enough tapes on the positive side.

    Tapes are the binding count, not clips. Cuts from one tape are near duplicates,
    so a type carried by two recordings has an effective sample size of two however
    many clips it spans.

    Viability is judged after the call type's own guard is applied, so a type is
    counted on the clips it will actually be trained on.
    """
    tasks = []
    for column in ann.call_columns(labels):
        call_type = column.removeprefix(CALL_PREFIX)
        guard = vocabulary.guard_for(call_type) if vocabulary else None
        eligible = labels if guard is None else labels[labels["duration_seconds"] <= guard]

        positive = eligible[eligible[column].fillna(False)]
        tapes = positive["tape_id"].nunique()
        if len(positive) >= MINIMUM_CLIPS and tapes >= MINIMUM_TAPES:
            tasks.append(Task(species, call_type, len(positive), tapes, guard))
    if not tasks:
        raise CallTypeError(
            f"no call type in {species} reaches {MINIMUM_CLIPS} clips over {MINIMUM_TAPES} tapes"
        )
    return sorted(tasks, key=lambda task: -task.tapes)


def _window_index(
    base: FeatureSource, labels: pd.DataFrame, task: Task
) -> tuple[pd.DataFrame, np.ndarray]:
    """Windows of this species, relabelled by whether the call type is present.

    A guard declared against this call type drops the clips too long for their own
    label to describe a window of them.
    """
    if task.max_clip_seconds is not None:
        labels = labels[labels["duration_seconds"] <= task.max_clip_seconds]

    flags = labels.set_index("clip_id")[f"{CALL_PREFIX}{task.call_type}"].fillna(False)
    index = base.index
    positions = np.flatnonzero(index["clip_id"].isin(set(flags.index)).to_numpy())

    subset = index.iloc[positions].reset_index(drop=True)
    present = subset["clip_id"].map(flags).astype(bool)
    subset = subset.assign(
        label=present.astype(int),
        species=np.where(present, PRESENT, ABSENT),
    )
    return subset, positions


def _context_index(subset: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The same windows, described by where and under what conditions they were made."""
    columns = [
        "site",
        "latitude",
        "longitude",
        "collection_code",
        *[c for c in labels.columns if c.startswith("cond_")],
    ]
    return subset.merge(labels[["clip_id", *columns]], on="clip_id", how="left")


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
    subset, positions = _window_index(base, labels, task)
    folds = folds_for_index(subset, cfg)
    plan = FoldPlan.repeated(cfg, subset, repeats) if repeats > 1 else FoldPlan.single(folds)

    audio = DerivedSource(base, subset, positions, name=f"{base.name}_{task.call_type}")
    context_index = _context_index(subset, labels)
    control = ContextFeatureSource(
        context_index, [c for c in context_index.columns if c.startswith("cond_")]
    )

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
