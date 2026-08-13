"""What a result directory name refers to.

Results are named after the question rather than after the model, so
``calltype_spermwhale_coda_cnn_small`` says which species, which call and which model
all in one string. Anything reading a finished result back has to take that name apart
again: which registry entry fitted it, which windows it saw, what its labels mean.

That resolution used to live beside the explanations that needed it, which gave one
module two reasons to change and left it importing the training stack to answer a
question about a filename. The grammar itself belongs to ``ResultName`` in
``src.results``; this is the part that turns a parsed name back into objects.
"""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.data.notes import load_vocabulary
from src.errors import DaturaError
from src.evaluate import families
from src.features import registry as features
from src.features.source import DerivedSource, FeatureSource
from src.models import registry as models
from src.models.registry import ModelSpec
from src.results import ResultName
from src.train import tasks

# What fitted a call type result whose directory names no model. run_task leaves the
# default untagged, so the bare name means trees.
DEFAULT_CALL_TYPE_MODEL = "xgboost"


class ResolveError(DaturaError):
    """Raised when a result name does not refer to anything that can be rebuilt."""


def spec_for_result(name: str) -> ModelSpec:
    """The registry entry behind a result directory.

    The model is whichever registry name the directory ends with; a call type result
    with no such suffix was fitted by the default tree model.
    """
    if name in models.names():
        return models.get(name)

    for candidate in sorted(models.names(), key=len, reverse=True):
        if name.endswith(f"_{candidate}"):
            return models.get(candidate)
    return models.get(DEFAULT_CALL_TYPE_MODEL)


def source_for_result(cfg: Config, spec: ModelSpec, name: str) -> FeatureSource:
    """The windows this result was fitted on, rebuilt.

    A species model saw the whole cache. A call type model saw a relabelled subset of
    it, so reading one back against the full cache would feed it windows it was never
    shown and score it on folds it never had. The task is posed again from the same
    inputs rather than stored, which is what keeps the two in step.
    """
    base = features.load_source(spec.source, cfg)
    if not name.startswith(families.CALL_TYPE_PREFIX):
        return base

    species, call_type = task_of(cfg, name)
    labels = tasks.clip_labels(cfg, species)
    guard = load_vocabulary().guard_for(call_type)
    task = tasks.Task(species, call_type, 0, 0, guard)

    subset, positions = tasks.window_index(base, labels, task)
    return DerivedSource(base, subset, positions, name=f"{base.name}_{call_type}")


def task_of(cfg: Config, name: str) -> tuple[str, str]:
    """The species and call type a result directory is named after."""
    try:
        parsed = ResultName.parse(name, species=cfg.dataset.species, models=models.names())
        return parsed.task
    except ValueError as error:
        raise ResolveError(str(error)) from error


def class_names_for_result(cfg: Config, name: str) -> list[str]:
    """What this result's labels mean, read from the result rather than assumed.

    The species set was hardcoded once, so a call type model could not be explained
    even though the occlusion test never cared what the classes were. Every result
    writes its classes into its own confusion matrix, which is where families reads
    them from too.
    """
    try:
        return list(families.class_names_of(cfg, name))
    except families.FamilyError:
        return list(cfg.dataset.species)


def frequency_axis(cfg: Config, spec: ModelSpec) -> np.ndarray:
    """The frequency of each row of this model's input, for labelling a band.

    Read off the representation the spec declares rather than assumed to be log mel.
    A flat descriptor vector has no frequency axis, so occlusion by band means nothing
    there and the error says which representation was asked.
    """
    extractor = features.build_extractor(spec.source, cfg)
    axis = getattr(extractor, "mel_frequencies", None)
    if axis is None:
        raise ResolveError(
            f"{spec.source} features have no frequency axis, so a band cannot be masked"
        )
    return axis()
