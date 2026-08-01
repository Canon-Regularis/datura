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
from typing import Any

from src.config import Config, load_yaml
from src.errors import DaturaError
from src.features import registry as features
from src.models.base import WindowClassifier

METADATA_SOURCE = "metadata"

Builder = Callable[[Config, dict[str, Any]], WindowClassifier]

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


_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="xgboost",
        trainer=TREES,
        source=features.ACOUSTIC,
        config_file="configs/xgb.yaml",
        build=_build_trees,
        summary="gradient boosted trees over hand engineered descriptors",
    ),
    ModelSpec(
        name="cnn",
        trainer=NETWORK,
        source=features.LOGMEL,
        config_file="configs/cnn.yaml",
        build=_build_cnn,
        summary="residual CNN over log mel windows",
    ),
    ModelSpec(
        name="cnn_small",
        trainer=NETWORK,
        source=features.LOGMEL,
        config_file="configs/cnn_small.yaml",
        build=_build_cnn,
        summary="the same network at a tenth of the capacity",
    ),
    ModelSpec(
        name=METADATA_SOURCE,
        trainer=TREES,
        source=METADATA_SOURCE,
        config_file="configs/metadata.yaml",
        build=_build_trees,
        summary="recording metadata only, the floor every audio model must clear",
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
    """The metadata control. Reports are written relative to it."""
    return next(spec for spec in _SPECS if spec.is_control)


def trained_by(trainer: str) -> tuple[ModelSpec, ...]:
    """Every model one entry point is responsible for."""
    return tuple(spec for spec in _SPECS if spec.trainer == trainer)


def audio_models() -> tuple[ModelSpec, ...]:
    """Everything that actually listens to the recording."""
    return tuple(spec for spec in _SPECS if not spec.is_control)
