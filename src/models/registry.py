"""The catalogue of models.

One declaration per model: what it is called, which features it consumes, and where
its hyperparameters live. The trainers, the pipeline and the report all read this
table, so adding a model is a single entry rather than an edit in four files.

Builders import their framework inside the call. Reading the roster is cheap enough
for the report to do it without pulling torch into the process.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import Config, ConfigError, load_yaml
from src.errors import DaturaError
from src.features import registry as features
from src.models.base import WindowClassifier

# Taken from the feature registry rather than spelled again, because a control is a
# feature source and ``ModelSpec.source`` names one namespace. These two used to be
# declared here, which made the field a union of two namespaces resolved by an if chain
# in a third module. The metadata control is the floor every published margin is
# measured from; the logbook sees the rest of the paperwork besides, and exists to say
# how much of that floor the paperwork was carrying.
METADATA_SOURCE = features.METADATA
LOGBOOK_SOURCE = features.LOGBOOK

Builder = Callable[[Config, dict[str, Any]], WindowClassifier]

# Reading a fitted model back off disk. Every model that writes a checkpoint declares
# one. The trees did not for a long time, on the reasoning that they refit in seconds,
# which held until a prediction command needed something to load.
Loader = Callable[[Path, dict[str, Any], int], WindowClassifier]

# Which command trains a model. The tree models share one, because the control has
# to be fitted on the same folds as the baseline it is compared against.
TREES = "trees"
NETWORK = "network"


class UnknownModel(DaturaError):
    """Raised when a model name has no entry in the registry."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown model {name!r}; expected one of {names()}")


# What each kind of hyperparameter file may contain. ``None`` means the block's keys
# are handed straight to the estimator, so enumerating them here would only duplicate
# its signature; the block is still required to exist. A named set means every key is
# read by this project's own code with an explicit default, which is where a typo used
# to disappear: misspell `epochs` in configs/cnn.yaml and the file said 30 while the
# run trained 40, with nothing printed either way.
TREE_SETTINGS: dict[str, frozenset[str] | None] = {"model": None}

TORCH_TRAIN_KEYS = frozenset(
    {
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "early_stopping_patience",
        "seed",
        "device",
        "deterministic",
    }
)

NETWORK_SETTINGS: dict[str, frozenset[str] | None] = {
    "model": None,
    "train": TORCH_TRAIN_KEYS,
    "augment": frozenset(
        {
            "enabled",
            "probability",
            "max_time_shift",
            "noise_std",
            "freq_mask_bins",
            "time_mask_frames",
        }
    ),
}

PROBE_SETTINGS: dict[str, frozenset[str] | None] = {"model": None, "train": TORCH_TRAIN_KEYS}


@dataclass(frozen=True)
class ModelSpec:
    """Everything the runner needs to train one model."""

    name: str
    source: str
    config_file: str
    trainer: str
    build: Builder
    summary: str
    settings_schema: dict[str, frozenset[str] | None] = field(default_factory=dict)
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
        return not features.is_control(self.source)

    @property
    def is_control(self) -> bool:
        """Whether this model is the floor the audio results are measured against."""
        return self.source == METADATA_SOURCE


def _build_trees(cfg: Config, settings: dict[str, Any]) -> WindowClassifier:
    from src.models.gbdt import GradientBoostedTrees

    params = dict(settings["model"])
    params.setdefault("random_state", cfg.split.seed)
    return GradientBoostedTrees(params)


def _load_trees(checkpoint: Path, settings: dict[str, Any], n_classes: int) -> WindowClassifier:
    from src.models.gbdt import GradientBoostedTrees

    return GradientBoostedTrees.load(checkpoint, settings["model"], n_classes)


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
        settings_schema=TREE_SETTINGS,
        build=_build_trees,
        load=_load_trees,
        repeats=10,
        summary="gradient boosted trees over hand engineered descriptors",
    ),
    ModelSpec(
        name="cnn",
        trainer=NETWORK,
        source=features.LOGMEL,
        config_file="configs/cnn.yaml",
        settings_schema=NETWORK_SETTINGS,
        build=_build_cnn,
        load=_load_cnn,
        summary="residual CNN over log mel windows",
    ),
    ModelSpec(
        name="cnn_small",
        trainer=NETWORK,
        source=features.LOGMEL,
        config_file="configs/cnn_small.yaml",
        settings_schema=NETWORK_SETTINGS,
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
        settings_schema=TREE_SETTINGS,
        build=_build_trees,
        load=_load_trees,
        repeats=10,
        summary="recording metadata only, the floor every audio model must clear",
    ),
    ModelSpec(
        name=LOGBOOK_SOURCE,
        trainer=TREES,
        source=LOGBOOK_SOURCE,
        config_file="configs/logbook.yaml",
        settings_schema=TREE_SETTINGS,
        build=_build_trees,
        load=_load_trees,
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
        settings_schema=PROBE_SETTINGS,
        build=_build_probe,
        load=_load_probe,
        repeats=10,
        summary="a linear probe over frozen pretrained encoder embeddings",
    ),
    # The same trees on the same descriptors, with each recording's mean subtracted
    # first. It is the only change measured here that moves an audio score upward, and
    # it moves the place held out score more than twice as far as the tape held out one.
    # It shares configs/xgb.yaml, because the point is that nothing but the input moved.
    ModelSpec(
        name="xgboost_centred",
        trainer=TREES,
        source=features.ACOUSTIC_CENTRED,
        config_file="configs/xgb.yaml",
        settings_schema=TREE_SETTINGS,
        build=_build_trees,
        load=_load_trees,
        repeats=10,
        summary="acoustic descriptors with the recording's mean removed",
    ),
)

_BY_NAME = {spec.name: spec for spec in _SPECS}


def load_settings(spec: ModelSpec, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read a model's hyperparameters, applying any command line overrides.

    Hyperparameters live in YAML beside the model that uses them, so a variant is a
    new config file and one registry entry.

    Validated against the spec's schema, because nothing checked these files and the
    consequence was silent. Every setting is read with an explicit default, so a
    misspelled key fell back to a value the file did not contain and the run carried on
    without a word. The four defaults in ``configs/cnn.yaml`` all disagree with the code
    they back onto: the file says 30 epochs and the fallback is 40, 64 against 32, 0.004
    against 0.003, 7 against 8.
    """
    settings = load_yaml(spec.config_file)
    for section, values in (overrides or {}).items():
        settings.setdefault(section, {}).update(values)
    _check_settings(spec, settings)
    return settings


def _check_settings(spec: ModelSpec, settings: dict[str, Any]) -> None:
    """Refuse a hyperparameter file this model does not understand."""
    if not spec.settings_schema:
        return

    unknown = set(settings) - set(spec.settings_schema)
    if unknown:
        raise ConfigError(
            f"unknown blocks in {spec.config_file}: {sorted(unknown)}; "
            f"expected some of {sorted(spec.settings_schema)}"
        )

    for block, allowed in spec.settings_schema.items():
        values = settings.get(block)
        if values is None:
            if allowed is None or block == "augment":
                continue  # augment is optional; the probe has none at all
            raise ConfigError(f"{spec.config_file} is missing the {block} block")
        if not isinstance(values, dict):
            raise ConfigError(f"{block} in {spec.config_file} must be a mapping")
        if allowed is None:
            continue  # passed straight to the estimator, which validates its own names
        stray = set(values) - allowed
        if stray:
            raise ConfigError(
                f"unknown keys in {block} of {spec.config_file}: {sorted(stray)}; "
                f"expected some of {sorted(allowed)}"
            )


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
