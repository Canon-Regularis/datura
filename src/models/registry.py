"""The catalogue of models.

One declaration per model: what it is called, which features it consumes, and where
its hyperparameters live. The trainers, the pipeline and the report all read this
table, so adding a model is a single entry rather than an edit in four files.

Builders import their framework inside the call. Reading the roster is cheap enough
for the report to do it without pulling torch into the process.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Config, load_yaml
from src.errors import DaturaError
from src.features import registry as features
from src.models.base import WindowClassifier

METADATA_SOURCE = "metadata"
LOGBOOK_SOURCE = "logbook"

# Neither of these hears the recording. The metadata control is the floor every
# published margin is measured from; the logbook sees the rest of the paperwork
# besides, and exists to say how much of the floor that paperwork was carrying.
NO_AUDIO_SOURCES = (METADATA_SOURCE, LOGBOOK_SOURCE)

Builder = Callable[[Config, dict[str, Any]], WindowClassifier]

# Reading a fitted model back off disk. Only the models that write a checkpoint
# declare one, which is why it is optional: the trees are refitted in seconds and
# never saved, so nothing can load them and nothing needs to.
Loader = Callable[[Path, dict[str, Any], int], WindowClassifier]

# Which command trains a model. The tree models share one, because the control has
# to be fitted on the same folds as the baseline it is compared against.
TREES = "trees"
NETWORK = "network"


class UnknownModel(DaturaError):
    """Raised when a model name has no entry in the registry."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown model {name!r}; expected one of {names()}")


@dataclass(frozen=True)
class ModelSpec:
    """Everything the runner needs to train one model."""

    name: str
    source: str
    config_file: str
    trainer: str
    build: Builder
    summary: str
    load: Loader | None = None
    repeats: int = 1
    """How many times the whole split is redrawn for this model.

    A cost decision, so it belongs beside the model it costs. The trees refit in
    seconds and the probe fits a linear map over cached vectors, so both can afford
    ten repeats of the five fold split and the fifty estimates that buys. The larger
    network is an hour and a half a run, and ten of those is most of a day of GPU for
    a comparison that is already the weakest in the report.
    """

    @property
    def hears_audio(self) -> bool:
        """Whether this model is given the recording at all."""
        return self.source not in NO_AUDIO_SOURCES

    @property
    def is_control(self) -> bool:
        """Whether this model is the floor the audio results are measured against."""
        return self.source == METADATA_SOURCE


def _build_trees(cfg: Config, settings: dict[str, Any]) -> WindowClassifier:
    from src.models.gbdt import GradientBoostedTrees

    params = dict(settings["model"])
    params.setdefault("random_state", cfg.split.seed)
    return GradientBoostedTrees(params)


def _build_cnn(cfg: Config, settings: dict[str, Any]) -> WindowClassifier:
    from src.models.cnn import SpectrogramCNN

    train = dict(settings["train"])
    train.setdefault("seed", cfg.split.seed)
    return SpectrogramCNN(settings["model"], train, settings["augment"])


def _load_cnn(checkpoint: Path, settings: dict[str, Any], n_classes: int) -> WindowClassifier:
    from src.models.cnn import SpectrogramCNN

    return SpectrogramCNN.load(checkpoint, settings["train"], n_classes)


def _build_probe(cfg: Config, settings: dict[str, Any]) -> WindowClassifier:
    from src.models.probe import EmbeddingProbe

    train = dict(settings["train"])
    train.setdefault("seed", cfg.split.seed)
    return EmbeddingProbe(settings["model"], train)


def _load_probe(checkpoint: Path, settings: dict[str, Any], n_classes: int) -> WindowClassifier:
    from src.models.probe import EmbeddingProbe

    return EmbeddingProbe.load(checkpoint, settings["train"], n_classes)


_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="xgboost",
        trainer=TREES,
        source=features.ACOUSTIC,
        config_file="configs/xgb.yaml",
        build=_build_trees,
        repeats=10,
        summary="gradient boosted trees over hand engineered descriptors",
    ),
    ModelSpec(
        name="cnn",
        trainer=NETWORK,
        source=features.LOGMEL,
        config_file="configs/cnn.yaml",
        build=_build_cnn,
        load=_load_cnn,
        summary="residual CNN over log mel windows",
    ),
    ModelSpec(
        name="cnn_small",
        trainer=NETWORK,
        source=features.LOGMEL,
        config_file="configs/cnn_small.yaml",
        build=_build_cnn,
        load=_load_cnn,
        repeats=10,
        summary="the same network at a tenth of the capacity",
    ),
    ModelSpec(
        name=METADATA_SOURCE,
        trainer=TREES,
        source=METADATA_SOURCE,
        config_file="configs/metadata.yaml",
        build=_build_trees,
        repeats=10,
        summary="recording metadata only, the floor every audio model must clear",
    ),
    ModelSpec(
        name=LOGBOOK_SOURCE,
        trainer=TREES,
        source=LOGBOOK_SOURCE,
        config_file="configs/logbook.yaml",
        build=_build_trees,
        repeats=10,
        summary="everything written down about a recording, and none of the recording",
    ),
    # Appended rather than slotted beside the other audio models on purpose. The
    # order here sets the row order of every committed table, so inserting in the
    # middle would move rows that did not otherwise change.
    ModelSpec(
        name="probe",
        trainer=NETWORK,
        source=features.ENCODER,
        config_file="configs/probe.yaml",
        build=_build_probe,
        load=_load_probe,
        repeats=10,
        summary="a linear probe over frozen pretrained encoder embeddings",
    ),
)

_BY_NAME = {spec.name: spec for spec in _SPECS}


def load_settings(spec: ModelSpec, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read a model's hyperparameters, applying any command line overrides.

    Hyperparameters live in YAML beside the model that uses them, so a variant is a
    new config file and one registry entry.
    """
    settings = load_yaml(spec.config_file)
    for section, values in (overrides or {}).items():
        settings.setdefault(section, {}).update(values)
    return settings


def names() -> tuple[str, ...]:
    """Every model, in the order results should be reported."""
    return tuple(_BY_NAME)


def specs() -> tuple[ModelSpec, ...]:
    return _SPECS


def get(name: str) -> ModelSpec:
    if name not in _BY_NAME:
        raise UnknownModel(name)
    return _BY_NAME[name]


def control() -> ModelSpec:
    """The metadata control. Reports are written relative to it.

    Named rather than found, because there is now a second model that hears no
    audio and picking whichever came first would silently move every margin.
    """
    return _BY_NAME[METADATA_SOURCE]


def trained_by(trainer: str) -> tuple[ModelSpec, ...]:
    """Every model one entry point is responsible for."""
    return tuple(spec for spec in _SPECS if spec.trainer == trainer)
