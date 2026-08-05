"""What the network actually uses: band occlusion, then Grad-CAM.

Both run against a saved fold checkpoint, so the model being explained is the one
that produced the reported score for that fold, on tapes it never saw.

Occlusion measures what the model loses when a band disappears; Grad-CAM shows
where the last convolutional stage responded. The first is a cost the model has to
pay, the second is a lead worth following, so they are reported together.

Usage:
    python -m src.evaluate.explain [--config configs/base.yaml] [--name cnn_small] [--fold 3]
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from src import cli
from src.config import Config
from src.data.notes import load_vocabulary
from src.data.splits import Fold, folds_for_index, rows_for_clips
from src.errors import DaturaError
from src.evaluate import families, plots
from src.evaluate.gradcam import GradCam
from src.evaluate.occlusion import band_occlusion
from src.features import registry as features
from src.features.source import DerivedSource, FeatureSource
from src.models import registry as models
from src.models.base import WindowClassifier
from src.models.registry import ModelSpec, load_settings
from src.results import checkpoint_path, model_directory
from src.train import tasks

logger = logging.getLogger(__name__)

EXAMPLES_PER_SPECIES = 2

# What fitted a call type result whose directory names no model. run_task leaves the
# default untagged, so the bare name means trees.
DEFAULT_CALL_TYPE_MODEL = "xgboost"


class ExplainError(DaturaError):
    """Raised when there is no trained model to explain."""


def load_fold_model(
    cfg: Config, spec: ModelSpec, settings: dict, fold_index: int, n_classes: int, name: str
) -> WindowClassifier:
    """Read one fold's weights back, through whichever model wrote them.

    ``name`` is the result directory rather than the registry entry, so a call type
    network is loaded by the same path as a species one: the checkpoint is keyed on
    the result, and the spec only says how to read it.
    """
    if spec.load is None:
        raise ExplainError(
            f"{spec.name} writes no checkpoint, so there is nothing to explain; "
            "the trees are refitted in seconds and never saved"
        )

    checkpoint = checkpoint_path(cfg, name, fold_index)
    if not checkpoint.with_suffix(".pt").exists():
        raise ExplainError(
            f"no checkpoint at {checkpoint.with_suffix('.pt')}; "
            f"run python -m src.train.cnn --name {name} first"
        )
    return spec.load(checkpoint, settings, n_classes)


def sample_correct_windows(
    model: WindowClassifier,
    source: FeatureSource,
    rows: np.ndarray,
    class_names: list[str],
    seed: int,
    per_species: int = EXAMPLES_PER_SPECIES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pick correctly classified windows, so the maps show evidence that worked."""
    rng = np.random.default_rng(seed)
    view = source.matrix(rows)
    predicted = model.predict_proba(view).argmax(axis=1)
    truth = source.index.iloc[rows]["label"].to_numpy()

    positions: list[int] = []
    labels: list[str] = []
    for label, name in enumerate(class_names):
        correct = np.flatnonzero((truth == label) & (predicted == label))
        pool = correct if len(correct) else np.flatnonzero(truth == label)
        if not len(pool):
            continue
        chosen = rng.choice(pool, size=min(per_species, len(pool)), replace=False)
        positions.extend(chosen.tolist())
        labels.extend(f"{name} (predicted {class_names[predicted[p]]})" for p in chosen.tolist())

    windows = view.take(np.asarray(positions, dtype=np.int64))
    heatmaps, _ = GradCam(model).heatmaps(windows)
    return windows, heatmaps, labels


def peak_frequencies(heatmaps: np.ndarray, frequencies: np.ndarray, labels: list[str]):
    """Where each map put most of its weight, in hertz."""
    return pd.DataFrame(
        {
            "example": labels,
            "peak_frequency_hz": [
                float(frequencies[int(np.argmax(h.mean(axis=1)))]) for h in heatmaps
            ],
        }
    )


def _class_names(cfg: Config, name: str) -> list[str]:
    """What this result's labels mean, read from the result rather than assumed.

    The species set was hardcoded here, so a call type model could not be explained
    even though the occlusion test never cared what the classes were. Every result
    writes its classes into its own confusion matrix, which is where families reads
    them from too.
    """
    try:
        return list(families.class_names_of(cfg, name))
    except families.FamilyError:
        return list(cfg.dataset.species)


def _frequencies(cfg: Config, spec: ModelSpec) -> np.ndarray:
    """The frequency of each row of this model's input, for labelling a band.

    Read off the representation the spec declares rather than assumed to be log mel.
    A flat descriptor vector has no frequency axis, so occlusion by band means
    nothing there and the error says which representation was asked.
    """
    extractor = features.build_extractor(spec.source, cfg)
    axis = getattr(extractor, "mel_frequencies", None)
    if axis is None:
        raise ExplainError(
            f"{spec.source} features have no frequency axis, so a band cannot be masked"
        )
    return axis()


def run(
    cfg: Config,
    spec: ModelSpec,
    settings: dict,
    fold: Fold,
    source: FeatureSource,
    name: str,
) -> None:
    class_names = _class_names(cfg, name)
    directory = model_directory(cfg, name)
    model = load_fold_model(cfg, spec, settings, fold.index, len(class_names), name)
    rows = rows_for_clips(source.index, fold.test_clips)
    logger.info("explaining %s fold %d on %d held out windows", name, fold.index, len(rows))

    frequencies = _frequencies(cfg, spec)

    table = band_occlusion(model, source.matrix(rows), source.index, rows, class_names, frequencies)
    table.insert(0, "fold", fold.index)
    table.to_csv(directory / "occlusion.csv", index=False)
    plots.occlusion_profile(table, directory / "occlusion.png", class_names)
    logger.info(
        "\nMacro-F1 lost per masked band\n%s",
        table[["band_low_hz", "band_high_hz", "macro_f1", "macro_f1_drop"]]
        .round(3)
        .to_string(index=False),
    )

    windows, heatmaps, labels = sample_correct_windows(
        model, source, rows, class_names, cfg.split.seed
    )
    plots.gradcam_panel(
        windows,
        heatmaps,
        labels,
        directory / "gradcam.png",
        frequencies=frequencies,
        seconds=cfg.audio.window_seconds,
    )
    peak_frequencies(heatmaps, frequencies, labels).to_csv(
        directory / "gradcam_peaks.csv", index=False
    )
    logger.info("figures written to %s", directory)


def spec_for_result(name: str) -> ModelSpec:
    """The registry entry behind a result directory.

    A result is named after the question rather than after the model, so a call type
    network lands in ``calltype_spermwhale_coda_cnn_small``. The model is whichever
    registry name the directory ends with; a call type result with no such suffix was
    fitted by the default tree model, which writes no checkpoint and will say so.
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
    it, so explaining one against the full cache would feed it windows it was never
    shown and score it on folds it never had. The task is posed again from the same
    inputs rather than stored, which is what keeps the two in step.
    """
    base = features.load_source(spec.source, cfg)
    if not name.startswith(families.CALL_TYPE_PREFIX):
        return base

    species, call_type = _task_of(cfg, name)
    labels = tasks.clip_labels(cfg, species)
    guard = load_vocabulary().guard_for(call_type)
    task = tasks.Task(species, call_type, 0, 0, guard)

    subset, positions = tasks.window_index(base, labels, task)
    return DerivedSource(base, subset, positions, name=f"{base.name}_{call_type}")


def _task_of(cfg: Config, name: str) -> tuple[str, str]:
    """The species and call type a result directory is named after."""
    body = name.removeprefix(families.CALL_TYPE_PREFIX)
    for species in cfg.dataset.species:
        prefix = f"{species.lower()}_"
        if not body.startswith(prefix):
            continue
        call_type = body.removeprefix(prefix)
        for candidate in models.names():
            call_type = call_type.removesuffix(f"_{candidate}")
        return species, call_type
    raise ExplainError(f"{name} names no species in this configuration")


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    cli.add_variant_name(parser, default="cnn_small")
    parser.add_argument("--fold", type=int, default=0, help="which fold's checkpoint to explain")
    args = parser.parse_args(argv)

    cfg = cli.prepare(args)
    spec = spec_for_result(args.name)
    source = source_for_result(cfg, spec, args.name)
    folds = folds_for_index(source.index, cfg)
    if not 0 <= args.fold < len(folds):
        raise ExplainError(f"fold {args.fold} is outside 0..{len(folds) - 1}")

    run(cfg, spec, load_settings(spec), folds[args.fold], source, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
