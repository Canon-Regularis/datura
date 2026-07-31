"""Frequency band occlusion.

Hide one band of the spectrogram from a trained model and measure what the score
loses. Unlike a saliency map this is a claim the model has to pay for: if masking
900 to 1800 Hz costs nothing, the model was not using it, whatever the heatmap
suggested.

Bands are reported by their centre frequency so the profile can be read against
known call bands for each species.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from src.evaluate import metrics
from src.features.source import MaskedRowView, RowView
from src.models.base import WindowClassifier


def band_edges(n_mels: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal groups of mel bins covering the whole axis."""
    cuts = np.linspace(0, n_mels, n_groups + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in pairwise(cuts) if b > a]


def _mask_for(row_shape: tuple[int, ...], low: int, high: int) -> np.ndarray:
    mask = np.ones(row_shape, dtype=np.float32)
    mask[low:high, :] = 0.0
    return mask


def _evaluate(
    model: WindowClassifier,
    view: RowView,
    index: pd.DataFrame,
    rows: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    probabilities = model.predict_proba(view)
    clips = metrics.aggregate_to_clips(index, rows, probabilities)
    clip_probabilities = clips[[f"p{i}" for i in range(len(class_names))]].to_numpy()
    return metrics.score(clips["label"].to_numpy(), clip_probabilities, class_names)


def band_occlusion(
    model: WindowClassifier,
    source_matrix: RowView,
    index: pd.DataFrame,
    rows: np.ndarray,
    class_names: list[str],
    mel_frequencies: np.ndarray,
    n_groups: int = 8,
) -> pd.DataFrame:
    """Score the model once per masked band and report the drop from baseline."""
    row_shape = source_matrix.row_shape
    if len(row_shape) != 2:
        raise ValueError(f"occlusion needs image-shaped rows, got {row_shape}")
    n_mels = row_shape[0]
    if len(mel_frequencies) != n_mels:
        raise ValueError(f"{len(mel_frequencies)} band frequencies for {n_mels} mel bins")

    baseline = _evaluate(model, source_matrix, index, rows, class_names)

    records = []
    for low, high in band_edges(n_mels, n_groups):
        masked = MaskedRowView(source_matrix, _mask_for(row_shape, low, high))
        scores = _evaluate(model, masked, index, rows, class_names)
        record = {
            "band_low_bin": low,
            "band_high_bin": high,
            "band_low_hz": float(mel_frequencies[low]),
            "band_high_hz": float(mel_frequencies[high - 1]),
            "band_center_hz": float(np.sqrt(mel_frequencies[low] * mel_frequencies[high - 1])),
            "macro_f1": scores["macro_f1"],
            "macro_f1_drop": baseline["macro_f1"] - scores["macro_f1"],
            "accuracy_drop": baseline["accuracy"] - scores["accuracy"],
        }
        for name in class_names:
            record[f"recall_{name}"] = scores[f"recall_{name}"]
            record[f"recall_{name}_drop"] = baseline[f"recall_{name}"] - scores[f"recall_{name}"]
        records.append(record)

    table = pd.DataFrame(records)
    table.attrs["baseline_macro_f1"] = baseline["macro_f1"]
    return table
