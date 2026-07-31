"""Fold construction. This is the only module that knows the grouping rule.

Cuts from one tape are near duplicates of each other, so splitting clips at random
puts copies of the same recording on both sides of the evaluation boundary and
produces accuracy figures that mean nothing. Folds are built over tapes, and every
model in the project calls this module rather than rolling its own split, so the
protection cannot drift between experiments.

Humpback whale survives on roughly a dozen tapes, which is why the project reports
a spread across folds instead of a single held-out score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import Config


class SplitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold, addressed by clip id rather than row position."""

    index: int
    train_clips: tuple[str, ...]
    validation_clips: tuple[str, ...]
    test_clips: tuple[str, ...]

    @property
    def fitting_clips(self) -> tuple[str, ...]:
        """Clips a model may look at while fitting, training plus validation."""
        return self.train_clips + self.validation_clips


def tape_id_of(clip_id: str, tape_id_length: int) -> str:
    """``5401800A`` and ``54018001`` are both cuts from tape ``54018``."""
    if len(clip_id) < tape_id_length:
        raise SplitError(f"clip id {clip_id!r} is shorter than the tape id length")
    return clip_id[:tape_id_length]


def _check_manifest(manifest: pd.DataFrame) -> None:
    required = {"clip_id", "tape_id", "label", "species"}
    missing = required - set(manifest.columns)
    if missing:
        raise SplitError(f"manifest is missing columns: {sorted(missing)}")
    if manifest["clip_id"].duplicated().any():
        duplicates = manifest.loc[manifest["clip_id"].duplicated(), "clip_id"].head().tolist()
        raise SplitError(f"clip ids must be unique, found duplicates such as {duplicates}")


def _grouped_holdout(
    frame: pd.DataFrame, n_splits: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split one frame into a larger and smaller part without breaking a tape apart."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    major, minor = next(
        iter(splitter.split(frame, frame["label"].to_numpy(), groups=frame["tape_id"].to_numpy()))
    )
    return major, minor


def make_folds(manifest: pd.DataFrame, cfg: Config, *, validation_splits: int = 5) -> list[Fold]:
    """Build ``cfg.split.n_folds`` folds grouped by tape and stratified by species.

    Each fold's fitting data is split again, still by tape, to give the CNN an
    early-stopping signal that the test tapes never touch.
    """
    _check_manifest(manifest)
    frame = manifest.reset_index(drop=True)
    labels = frame["label"].to_numpy()
    groups = frame["tape_id"].to_numpy()

    n_tapes = len(np.unique(groups))
    if n_tapes < cfg.split.n_folds:
        raise SplitError(
            f"{n_tapes} tapes cannot fill {cfg.split.n_folds} folds; lower split.n_folds"
        )

    splitter = StratifiedGroupKFold(
        n_splits=cfg.split.n_folds, shuffle=True, random_state=cfg.split.seed
    )

    folds: list[Fold] = []
    for index, (fit_rows, test_rows) in enumerate(splitter.split(frame, labels, groups)):
        fitting = frame.iloc[fit_rows]
        train_rows, validation_rows = _grouped_holdout(
            fitting, validation_splits, cfg.split.seed + index
        )
        folds.append(
            Fold(
                index=index,
                train_clips=tuple(fitting.iloc[train_rows]["clip_id"]),
                validation_clips=tuple(fitting.iloc[validation_rows]["clip_id"]),
                test_clips=tuple(frame.iloc[test_rows]["clip_id"]),
            )
        )

    assert_no_tape_leak(frame, folds, cfg.split.tape_id_length)
    return folds


def assert_no_tape_leak(manifest: pd.DataFrame, folds: list[Fold], tape_id_length: int) -> None:
    """Fail loudly if any tape appears on both sides of a fold boundary."""
    for fold in folds:
        parts = {
            "train": fold.train_clips,
            "validation": fold.validation_clips,
            "test": fold.test_clips,
        }
        tapes = {
            name: {tape_id_of(clip, tape_id_length) for clip in clips}
            for name, clips in parts.items()
        }
        for left, right in (("train", "test"), ("validation", "test"), ("train", "validation")):
            shared = tapes[left] & tapes[right]
            if shared:
                raise SplitError(
                    f"fold {fold.index} leaks {len(shared)} tape(s) between {left} and {right}: "
                    f"{sorted(shared)[:5]}"
                )
        covered = set(fold.train_clips) | set(fold.validation_clips) | set(fold.test_clips)
        if covered != set(manifest["clip_id"]):
            raise SplitError(f"fold {fold.index} does not partition the manifest")


def fold_summary(manifest: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Clip and tape counts per species per fold, the table that shows how thin
    the scarcest class really is."""
    by_clip = manifest.set_index("clip_id")
    rows = []
    for fold in folds:
        for part, clips in (
            ("train", fold.train_clips),
            ("validation", fold.validation_clips),
            ("test", fold.test_clips),
        ):
            subset = by_clip.loc[list(clips)]
            for species, group in subset.groupby("species"):
                rows.append(
                    {
                        "fold": fold.index,
                        "part": part,
                        "species": species,
                        "clips": len(group),
                        "tapes": group["tape_id"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def clips_from_index(index: pd.DataFrame) -> pd.DataFrame:
    """Collapse a window index to one row per clip, ready for fold construction.

    Folds are built from the clips that actually produced features rather than from
    the manifest, so a clip that failed to decode cannot end up in a fold with no
    rows behind it.
    """
    columns = ["clip_id", "tape_id", "species", "label"]
    return index.drop_duplicates("clip_id")[columns].reset_index(drop=True)


def rows_for_clips(index: pd.DataFrame, clips: tuple[str, ...] | list[str]) -> np.ndarray:
    """Positions of every window belonging to the given clips.

    Models train on windows and are scored on clips, so this is the bridge between
    a fold, defined over clips, and the feature matrix, defined over windows.
    """
    if "clip_id" not in index.columns:
        raise SplitError("window index must carry a clip_id column")
    mask = index["clip_id"].isin(set(clips)).to_numpy()
    return np.flatnonzero(mask)
