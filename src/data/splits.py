"""Fold construction. This is the only module that knows the grouping rule.

Cuts from one tape are near duplicates of each other, so splitting clips at random
puts copies of the same recording on both sides of the evaluation boundary and
produces accuracy figures that mean nothing. Folds are built over tapes, and every
model in the project calls this module rather than rolling its own split, so the
protection cannot drift between experiments.

Humpback whale survives on roughly a dozen tapes, which is why the project reports
a spread across folds instead of a single held out score.

Nothing here knows how a group id is formed. The config names the column and the
manifest fills it, so the same code groups Watkins by tape and a collection of
continuous recordings by site and year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import Config
from src.errors import DaturaError

DEFAULT_GROUP_COLUMN = "tape_id"


class SplitError(DaturaError):
    """Raised when folds would leak a tape, or cannot be built at all."""


@dataclass(frozen=True)
class Fold:
    """One cross validation fold, addressed by clip id rather than row position."""

    index: int
    train_clips: tuple[str, ...]
    validation_clips: tuple[str, ...]
    test_clips: tuple[str, ...]

    @property
    def fitting_clips(self) -> tuple[str, ...]:
        """Clips a model may look at while fitting, training plus validation."""
        return self.train_clips + self.validation_clips


def _check_manifest(manifest: pd.DataFrame, group_column: str = DEFAULT_GROUP_COLUMN) -> None:
    required = {"clip_id", group_column, "label", "species"}
    missing = required - set(manifest.columns)
    if missing:
        raise SplitError(f"manifest is missing columns: {sorted(missing)}")
    if manifest["clip_id"].duplicated().any():
        duplicates = manifest.loc[manifest["clip_id"].duplicated(), "clip_id"].head().tolist()
        raise SplitError(f"clip ids must be unique, found duplicates such as {duplicates}")


def _grouped_holdout(
    frame: pd.DataFrame, n_splits: int, seed: int, group_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Split one frame into a larger and smaller part without breaking a group apart."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    major, minor = next(
        iter(
            splitter.split(frame, frame["label"].to_numpy(), groups=frame[group_column].to_numpy())
        )
    )
    return major, minor


def make_folds(manifest: pd.DataFrame, cfg: Config, *, validation_splits: int = 5) -> list[Fold]:
    """Build ``cfg.split.n_folds`` folds grouped by recording and stratified by class.

    The grouping column is named by the config rather than derived here. Watkins
    fills it from the tape; a collection of continuous recordings would fill it from
    the site and year. Either way this module never learns which collection it is
    looking at.

    Each fold's fitting data is split again, still by group, to give the network an
    early stopping signal that the test groups never touch.
    """
    group_column = cfg.split.group_column
    _check_manifest(manifest, group_column)
    frame = manifest.reset_index(drop=True)
    labels = frame["label"].to_numpy()
    groups = frame[group_column].to_numpy()

    n_groups = len(np.unique(groups))
    if n_groups < cfg.split.n_folds:
        raise SplitError(
            f"{n_groups} {group_column} groups cannot fill {cfg.split.n_folds} folds; "
            "lower split.n_folds"
        )

    splitter = StratifiedGroupKFold(
        n_splits=cfg.split.n_folds, shuffle=True, random_state=cfg.split.seed
    )

    folds: list[Fold] = []
    for index, (fit_rows, test_rows) in enumerate(splitter.split(frame, labels, groups)):
        fitting = frame.iloc[fit_rows]
        train_rows, validation_rows = _grouped_holdout(
            fitting, validation_splits, cfg.split.seed + index, group_column
        )
        folds.append(
            Fold(
                index=index,
                train_clips=tuple(fitting.iloc[train_rows]["clip_id"]),
                validation_clips=tuple(fitting.iloc[validation_rows]["clip_id"]),
                test_clips=tuple(frame.iloc[test_rows]["clip_id"]),
            )
        )

    assert_no_group_leak(frame, folds, group_column)
    return folds


def assert_no_group_leak(
    manifest: pd.DataFrame, folds: list[Fold], group_column: str = DEFAULT_GROUP_COLUMN
) -> None:
    """Fail loudly if any group appears on both sides of a fold boundary.

    The group is read from the manifest rather than derived again from the clip id,
    so this checks the grouping the folds were actually built on, whatever
    collection the manifest describes.
    """
    group_of = manifest.set_index("clip_id")[group_column]
    for fold in folds:
        parts = {
            "train": fold.train_clips,
            "validation": fold.validation_clips,
            "test": fold.test_clips,
        }
        groups = {
            name: set(group_of.loc[list(clips)]) if clips else set()
            for name, clips in parts.items()
        }
        for left, right in (("train", "test"), ("validation", "test"), ("train", "validation")):
            shared = groups[left] & groups[right]
            if shared:
                raise SplitError(
                    f"fold {fold.index} leaks {len(shared)} {group_column} group(s) between "
                    f"{left} and {right}: {sorted(shared)[:5]}"
                )
        covered = set(fold.train_clips) | set(fold.validation_clips) | set(fold.test_clips)
        if covered != set(manifest["clip_id"]):
            raise SplitError(f"fold {fold.index} does not partition the manifest")


def fold_summary(
    manifest: pd.DataFrame, folds: list[Fold], group_column: str = DEFAULT_GROUP_COLUMN
) -> pd.DataFrame:
    """Clip and group counts per class per fold, the table that shows how thin
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
                        "tapes": group[group_column].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def clips_from_index(index: pd.DataFrame, group_column: str = DEFAULT_GROUP_COLUMN) -> pd.DataFrame:
    """Collapse a window index to one row per clip, ready for fold construction.

    Folds are built from the clips that actually produced features rather than from
    the manifest, so a clip that failed to decode cannot end up in a fold with no
    rows behind it.
    """
    columns = ["clip_id", group_column, "species", "label"]
    return index.drop_duplicates("clip_id")[columns].reset_index(drop=True)


def folds_for_index(index: pd.DataFrame, cfg: Config) -> list[Fold]:
    """Folds over whichever clips a window index actually contains.

    Training and explainability both need this. Sharing it is what keeps the fold
    a model was scored on identical to the fold its explanation is computed on.
    """
    return make_folds(clips_from_index(index, cfg.split.group_column), cfg)


def rows_for_clips(index: pd.DataFrame, clips: tuple[str, ...] | list[str]) -> np.ndarray:
    """Positions of every window belonging to the given clips.

    Models train on windows and are scored on clips, so this is the bridge between
    a fold, defined over clips, and the feature matrix, defined over windows.
    """
    if "clip_id" not in index.columns:
        raise SplitError("window index must carry a clip_id column")
    mask = index["clip_id"].isin(set(clips)).to_numpy()
    return np.flatnonzero(mask)
