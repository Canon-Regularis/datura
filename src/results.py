"""Where results live on disk.

Every path under the report tree is built here. Training writes a checkpoint,
explainability reads it back, and the report walks the same directories: if any of
them spelled the layout out for itself, renaming a folder would break the others
silently.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.config import Config

# How a result directory is named. A species model is named after itself; a call type
# result is named after the question it answers, with the model appended only when it
# is not the default one.
CALL_TYPE_PREFIX = "calltype_"
CONTEXT_MARKER = "_context"

CHECKPOINTS = "checkpoints"

# Per model, inside its own directory.
SUMMARY_FILE = "summary.csv"
PROVENANCE_FILE = "provenance.json"
PREDICTIONS_FILE = "clip_predictions.parquet"
WINDOW_PREDICTIONS_FILE = "window_predictions.parquet"
CLIP_METRICS_FILE = "fold_metrics_clip.csv"
VALIDATION_METRICS_FILE = "fold_metrics_validation.csv"
WINDOW_METRICS_FILE = "fold_metrics_window.csv"
CONFUSION_FILE = "confusion.csv"
OCCLUSION_FILE = "occlusion.csv"

# Per configuration, beside the models.
REPORT_FILE = "REPORT.md"
COMPARISON_FILE = "comparison.csv"
COVERAGE_FILE = "coverage.csv"
AMBIGUITY_FILE = "ambiguity_breakdown.csv"
FAMILY_MARGINS_FILE = "family_margins.csv"
MARGIN_OVER_CONTROL_FILE = "margin_over_control.csv"


def config_directory(cfg: Config) -> Path:
    """Everything produced under one configuration."""
    return cfg.paths.reports / cfg.name


def model_directory(cfg: Config, model_name: str) -> Path:
    """One trained model's metrics, predictions and figures."""
    return config_directory(cfg) / model_name


def checkpoint_path(cfg: Config, model_name: str, fold_index: int, repeat: int = 0) -> Path:
    """The weights saved for one fold of one repeat.

    Repeat zero keeps the plain name, so a single split writes exactly where it
    always did and the explainability tools keep finding it. The suffix is added by
    the writer.
    """
    stem = f"fold{fold_index}" if repeat == 0 else f"repeat{repeat}_fold{fold_index}"
    return model_directory(cfg, model_name) / CHECKPOINTS / stem


def summary_path(cfg: Config, model_name: str) -> Path:
    return model_directory(cfg, model_name) / SUMMARY_FILE


def clip_metrics_path(cfg: Config, model_name: str) -> Path:
    """One row per split, which is what every comparison is paired on."""
    return model_directory(cfg, model_name) / CLIP_METRICS_FILE


def window_metrics_path(cfg: Config, model_name: str) -> Path:
    return model_directory(cfg, model_name) / WINDOW_METRICS_FILE


def confusion_path(cfg: Config, model_name: str) -> Path:
    return model_directory(cfg, model_name) / CONFUSION_FILE


def occlusion_path(cfg: Config, model_name: str) -> Path:
    """What a trained network loses when one frequency band is hidden from it."""
    return model_directory(cfg, model_name) / OCCLUSION_FILE


def provenance_path(cfg: Config, model_name: str | None = None) -> Path:
    """What produced a result: commit, versions, accelerator, config digests.

    One per model and one for the configuration as a whole, so a directory can always
    say what wrote it.
    """
    root = config_directory(cfg) if model_name is None else model_directory(cfg, model_name)
    return root / PROVENANCE_FILE


def coverage_path(cfg: Config) -> Path:
    """Accuracy against the share of clips a model was allowed to decline."""
    return config_directory(cfg) / COVERAGE_FILE


def diagnostics_path(cfg: Config) -> Path:
    """What the audio identifies other than the species, for one corpus.

    Keyed on the corpus rather than on the configuration, because these questions never
    ask about the fold rule. ``base_10k``, ``context_10k`` and ``context_shuffled_10k``
    read one manifest and one feature cache, so keying on the name would compute the
    same answer three times and invite the three copies to disagree.
    """
    return cfg.paths.reports / f"diagnostics_{cfg.corpus}.csv"


def validation_metrics_path(cfg: Config, model_name: str) -> Path:
    """Per fold scores on the rows held out for early stopping.

    Kept beside the test scores so that anything chosen by looking at a number can be
    shown to have been chosen here. A setting picked on the test folds is a setting
    picked on the figure being published.
    """
    return model_directory(cfg, model_name) / VALIDATION_METRICS_FILE


def comparison_path(cfg: Config) -> Path:
    return config_directory(cfg) / COMPARISON_FILE


def ambiguity_path(cfg: Config) -> Path:
    return config_directory(cfg) / AMBIGUITY_FILE


def family_margins_path(cfg: Config) -> Path:
    """Every comparison in one configuration, which the correction reads across all."""
    return config_directory(cfg) / FAMILY_MARGINS_FILE


def margin_over_control_path(cfg: Config) -> Path:
    return config_directory(cfg) / MARGIN_OVER_CONTROL_FILE


def audit_table_path(cfg: Config, stem: str) -> Path:
    """A corpus level audit table, keyed by corpus rather than by experiment.

    Two configurations sharing a corpus share these, which is the whole point of the
    corpus being a separate name. Building the filename by hand is what let one caller
    write ``audit_cross_species_tapes_{corpus}.csv`` while another read a different
    spelling of the same thing.
    """
    return cfg.paths.metadata / f"{stem}_{cfg.corpus}.csv"


def manifest_path(cfg: Config) -> Path:
    """The one description of what data exists, keyed by corpus."""
    return cfg.paths.metadata / f"manifest_{cfg.corpus}.parquet"


def predictions_path(cfg: Config, model_name: str) -> Path:
    return model_directory(cfg, model_name) / PREDICTIONS_FILE


def window_predictions_path(cfg: Config, model_name: str) -> Path:
    """Per window scores before they are averaged up to a clip.

    Nothing reads this yet. It carries ``window_index``, which is the only time
    coordinate the pipeline produces, so it is where anything about when in a
    recording a call happens would have to begin.
    """
    return model_directory(cfg, model_name) / WINDOW_PREDICTIONS_FILE


def fold_summary_path(cfg: Config) -> Path:
    return cfg.paths.reports / f"fold_summary_{cfg.name}.csv"


def report_path(cfg: Config) -> Path:
    return config_directory(cfg) / REPORT_FILE


def has_results(cfg: Config, model_name: str) -> bool:
    """Whether a model has been trained under this configuration."""
    return summary_path(cfg, model_name).exists()


def ensure(cfg: Config) -> Path:
    """Create the report tree for this configuration and return its root."""
    directory = config_directory(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass(frozen=True)
class ResultName:
    """The name of one result directory, taken apart.

    One grammar, and it used to have four parsers. ``train.tasks`` and
    ``train.calltypes`` built these names; ``families`` took them apart three times and
    ``explain`` twice, each with its own loop over the species list and its own rule for
    stripping a model suffix. Six sites, four parsers, and any disagreement between them
    would attach a result to the wrong control, which is a margin measured against the
    wrong thing with nothing to say so.

    Parsing needs to know the species under study and the model names, because both
    appear inside the string and neither is guessable from it: a call type is
    ``killerwhale_pulsed_call`` and only the species list says where the species ends.
    They are passed in rather than imported so this module keeps depending on nothing.
    """

    call_type: str | None = None
    species: str | None = None
    model: str | None = None
    is_control: bool = False
    suffix: str = ""

    @property
    def is_call_type(self) -> bool:
        return self.call_type is not None

    @property
    def task(self) -> tuple[str, str]:
        """The species and the call type, for a result that poses one.

        Both or neither. Every caller that wants one wants the other, and asking for
        them together is what lets the check live in one place.
        """
        if self.call_type is None or self.species is None:
            raise ValueError(f"{self.render()} is a species result and names no call type")
        return self.species, self.call_type

    @property
    def task_key(self) -> str:
        """The task this result belongs to, which is what a family is keyed on."""
        species, call_type = self.task
        return f"{CALL_TYPE_PREFIX}{species.lower()}_{call_type}"

    def render(self) -> str:
        """The directory name, which is the only place this string is built."""
        if not self.is_call_type:
            return f"{self.model}{self.suffix}"
        marker = CONTEXT_MARKER if self.is_control else ""
        model = f"_{self.model}" if self.model else ""
        return f"{self.task_key}{marker}{model}{self.suffix}"

    @classmethod
    def parse(cls, name: str, *, species: Iterable[str], models: Iterable[str]) -> ResultName:
        """Take a directory name apart, or say it is a plain model.

        The longest matching model name wins, because one registry name can end with
        another and the shorter one would claim a result it did not fit.
        """
        known = sorted(models, key=len, reverse=True)
        if not name.startswith(CALL_TYPE_PREFIX):
            return cls(model=name)

        body = name.removeprefix(CALL_TYPE_PREFIX)
        for candidate in species:
            prefix = f"{candidate.lower()}_"
            if not body.startswith(prefix):
                continue
            rest = body.removeprefix(prefix)

            model = next((entry for entry in known if rest.endswith(f"_{entry}")), None)
            if model is not None:
                rest = rest.removesuffix(f"_{model}")

            is_control = CONTEXT_MARKER in rest
            if is_control:
                rest = rest.split(CONTEXT_MARKER)[0]
            return cls(call_type=rest, species=candidate, model=model, is_control=is_control)

        raise ValueError(f"{name} names no species in this configuration")
