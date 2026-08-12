"""Which call type questions are worth asking, and of which clips.

Deciding this needs the manifest and the field notes and nothing else. It does not
need a model, a fold or an estimator, so it does not import one: asking which types
are viable used to pull the whole training stack into the process, which made the
question expensive to ask from a notebook or a shell.

Tapes are the binding count here, not clips. Cuts from one tape are near duplicates,
so a call type carried by two recordings has an effective sample size of two however
many clips it spans.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config
from src.data import annotations as ann
from src.data.manifest import load_manifest
from src.data.notes import CALL_PREFIX, Vocabulary
from src.errors import DaturaError
from src.features.source import FeatureSource
from src.results import ResultName

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
        """The result directory this task writes to.

        Rendered by ``ResultName`` rather than formatted here, because four places used
        to take this string apart again and any disagreement would attach a result to
        the wrong control.
        """
        return ResultName(call_type=self.call_type, species=self.species).render()

    @property
    def control_name(self) -> str:
        return ResultName(call_type=self.call_type, species=self.species, is_control=True).render()

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
    joined = ann.attach_context(
        manifest.merge(parsed[["clip_id", *columns]], on="clip_id", how="left"), parsed
    )
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


def window_index(
    base: FeatureSource, labels: pd.DataFrame, task: Task
) -> tuple[pd.DataFrame, np.ndarray]:
    """Windows of this species, relabelled by whether the call type is present.

    A guard declared against this call type drops the clips too long for their own
    label to describe a window of them.

    Returned alongside the positions it selected, so a caller can build a view over
    the same rows of the audio cache. Posing the task and fitting it are separate
    jobs: the explainability tools rebuild exactly this subset to read a trained
    model back against the windows it actually saw.
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
