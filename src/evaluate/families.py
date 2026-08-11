"""Grouping results into the sets that are actually comparable.

A score on its own says nothing here. Every model in this project was trained
beside a control that sees no audio, and the only number worth reading is the gap
between them. A family is one such set: some models on trial, and the control they
were all measured against.

Discovery works off what is on disk rather than off the model registry. The
registry knows the four species models and nothing about the call type work, which
is why nineteen of twenty three result directories were invisible in the report.
Anything that wrote a summary belongs to a family, or the report says so and fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import Config
from src.errors import DaturaError
from src.evaluate import ensemble
from src.models import registry as models
from src.results import SUMMARY_FILE, config_directory

# A call type result is named after its task, and its control carries this marker.
# The species control is named by the registry instead, because it predates the
# convention and is referenced by name in several places.
CONTEXT_MARKER = "_context"

SPECIES_FAMILY = "species"
CALL_TYPE_PREFIX = "calltype_"


class FamilyError(DaturaError):
    """Raised when a result has no control to be measured against."""


@dataclass(frozen=True)
class Family:
    """Models on trial, and the control that gives their scores a meaning."""

    key: str
    title: str
    members: tuple[str, ...]
    control: str
    class_names: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Every result in the family, control last."""
        return (*self.members, self.control)

    @property
    def floors(self) -> tuple[str, ...]:
        """Every result in the family that is never given the recording."""
        silent = {spec.name for spec in models.specs() if not spec.hears_audio}
        return tuple(name for name in self.names if name in silent)


def strongest_floor(cfg: Config, family: Family) -> str | None:
    """The highest scoring model in this family that hears no audio.

    A control is only a floor while nothing else that ignores the recording beats
    it, and the metadata control was built before anyone knew what else the
    paperwork carried. Where a family holds two such models, the higher one is the
    number an audio result actually has to clear, and the gap between them says how
    much of the floor was equipment and how much was everything else written down.

    ``None`` when the family has nothing better than the control it already names,
    which is every family that was never given a second one.
    """
    floors = family.floors
    if len(floors) < 2:
        return None

    scores = {name: _headline_score(cfg, name) for name in floors}
    best = max(scores, key=scores.get)
    return None if best == family.control else best


def _headline_score(cfg: Config, name: str, metric: str = "macro_f1") -> float:
    summary = pd.read_csv(config_directory(cfg) / name / SUMMARY_FILE)
    return float(summary.loc[summary["metric"] == metric, "mean"].iloc[0])


def result_names(cfg: Config) -> list[str]:
    """Every directory under this configuration that a model actually wrote."""
    root = config_directory(cfg)
    if not root.exists():
        return []
    return sorted(child.name for child in root.iterdir() if (child / SUMMARY_FILE).exists())


def class_names_of(cfg: Config, name: str) -> tuple[str, ...]:
    """What the labels of one result mean, read from the result itself.

    The confusion matrix is written with its classes as the index, so a call type
    result declares ``absent`` and ``present`` and a species result declares the
    species. Reading it here keeps the report from having to know which is which.
    """
    path = config_directory(cfg) / name / "confusion.csv"
    if not path.exists():
        raise FamilyError(f"{name} has no confusion.csv, so its classes are unknown")
    return tuple(str(value) for value in pd.read_csv(path, index_col=0).index)


def _title(cfg: Config, key: str) -> str:
    """A readable heading for a call type family.

    Task names are lowercased when they become directories, so the species is
    recovered by matching against the species under study rather than by guessing
    where the word ends.
    """
    body = key.removeprefix(CALL_TYPE_PREFIX)
    for species in cfg.dataset.species:
        prefix = f"{species.lower()}_"
        if body.startswith(prefix):
            return f"{species}, {body.removeprefix(prefix).replace('_', ' ')}"
    return body.replace("_", " ")


def _species_family(cfg: Config, names: list[str]) -> Family | None:
    """The species models, measured against the metadata control.

    Derived results join the family too. An average of two fitted models is a model a
    person can run, so it earns a margin and a place in the multiplicity correction
    rather than a footnote. It is recognised by its name rather than by a registry
    entry, because nothing trained it and a spec with no trainer would be a lie.
    """
    control = models.control().name
    known = set(models.names())
    members = tuple(name for name in models.names() if name != control and name in names)
    members += tuple(name for name in names if ensemble.is_derived(name, known))
    if not members:
        return None
    if control not in names:
        raise FamilyError(
            f"{', '.join(members)} have results but {control} does not; "
            "rerun python -m src.train.xgb without --skip-control"
        )
    return Family(
        key=SPECIES_FAMILY,
        title="Species",
        members=members,
        control=control,
        class_names=class_names_of(cfg, control),
    )


def _controls_by_key(names: list[str]) -> dict[str, str]:
    """The context control belonging to each call type task."""
    controls: dict[str, str] = {}
    for name in names:
        if CONTEXT_MARKER not in name:
            continue
        key = name.split(CONTEXT_MARKER)[0]
        if key in controls:
            raise FamilyError(
                f"{key} has two controls, {controls[key]} and {name}; one of them is a leftover run"
            )
        controls[key] = name
    return controls


def _assign(trials: list[str], keys: list[str]) -> dict[str, list[str]]:
    """Put each model on trial with the most specific task it belongs to.

    A model is named after its task, with the model appended when it is not the
    default. That makes one task's name a prefix of another's whenever one call type
    name extends another, and matching on the prefix alone would hand a sibling's
    models to the shorter task. The margin would then be taken against the wrong
    control and nothing would say so. The longest matching key wins instead.
    """
    members: dict[str, list[str]] = {key: [] for key in keys}
    by_length = sorted(keys, key=len, reverse=True)

    for name in trials:
        owner = next((key for key in by_length if name == key or name.startswith(f"{key}_")), None)
        if owner is not None:
            members[owner].append(name)
    return members


def _call_type_families(cfg: Config, names: list[str]) -> list[Family]:
    """One family per call type task, measured against its context control."""
    controls = _controls_by_key(names)
    trials = [name for name in names if CONTEXT_MARKER not in name]
    members = _assign(trials, list(controls))

    families = []
    for key, control in controls.items():
        if not members[key]:
            raise FamilyError(f"{control} is a control for {key}, which has no results")
        families.append(
            Family(
                key=key,
                title=_title(cfg, key),
                members=tuple(members[key]),
                control=control,
                class_names=class_names_of(cfg, control),
            )
        )
    return families


def discover(cfg: Config) -> list[Family]:
    """Every family with results on disk, species first.

    A result that belongs to no family is an error rather than an omission. It means
    a model was trained without the control that makes its score readable, and
    printing it beside the others would invite exactly the comparison the control
    exists to prevent.
    """
    names = result_names(cfg)
    families = [
        *[family for family in [_species_family(cfg, names)] if family],
        *_call_type_families(cfg, names),
    ]

    claimed: dict[str, str] = {}
    for family in families:
        for name in family.names:
            if name in claimed:
                raise FamilyError(
                    f"{name} belongs to both {claimed[name]} and {family.key}; "
                    "it would be measured against two different controls"
                )
            claimed[name] = family.key

    orphans = sorted(set(names) - set(claimed))
    if orphans:
        raise FamilyError(f"these results have no control to be measured against: {orphans}")
    return families
