"""The models that are given no audio at all.

Every published margin in this project is a distance from one of these. They read
columns off the window index, code the names as identities for a tree to split on,
and never see a sample of sound. Whatever one of them reaches is the floor an audio
result has to clear before it says anything about whales.

They live apart from the sources that carry audio because they answer a different
question. A cached source asks what the recording sounds like; these ask what was
written down about it, which turned out to be the better predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.source import FeatureSource
from src.features.views import RowView


class TabularFeatureSource(FeatureSource):
    """A control built from columns of the window index, with no audio.

    Every control in this project is the same shape: some names coded as identities
    for a tree to split on, some numbers taken as they are, and nothing that came
    out of the recording itself. Written once so the three of them cannot drift
    apart in how they treat a missing value or a category.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        *,
        name: str,
        categorical: list[str],
        numeric: list[str],
    ):
        missing = {*categorical, *numeric} - set(index.columns)
        if missing:
            raise ValueError(f"window index is missing {name} columns: {sorted(missing)}")

        self._index = index.reset_index(drop=True)
        self._name = name
        self._categorical = list(categorical)
        self._numeric = list(numeric)

        # One column per name rather than one number per column, and the reason is
        # worth the paragraph because getting it wrong cost this project a published
        # claim in both directions.
        #
        # A site or a collection code is a name. Coding it to its position in the
        # alphabet and handing a tree the integer makes it an ordinal, so the only
        # question a split can ask is whether the code is above some threshold, and the
        # answer for a name the model never saw is decided by where the alphabet
        # happened to put it. Under tape folds that barely shows, because a held out
        # tape almost always carries a name the training tapes carried too. Under place
        # folds every held out name is new, and the arbitrary geometry is the whole
        # answer: five encodings carrying identical information scored 0.7211, 0.9908,
        # 0.9911, 0.9990 and 0.9993 on the same folds. Moving the absent name sentinel
        # from one end of the axis to the other, which carries no information at all,
        # was worth 0.278.
        #
        # A column per name removes the axis. Nothing is adjacent to anything, a name
        # the model never saw is zero everywhere, and the tree falls back on what it
        # does know rather than on a neighbour it was handed by accident. The same
        # comparison under this encoding spans 0.018.
        names = [
            pd.get_dummies(self._present(column), prefix=column, dtype=np.float32)
            for column in self._categorical
        ]
        self._name_columns = [list(block.columns) for block in names]

        # A missing number stays missing. Writing 0.0 for an absent coordinate invents
        # a measurement, and XGBoost takes NaN natively by learning a default direction
        # per split, which is the honest reading of a value nobody wrote down.
        numbers = self._index.loc[:, self._numeric].astype(float)
        blocks = [block.to_numpy() for block in names]
        self._matrix = np.column_stack([*blocks, numbers.to_numpy()]).astype(np.float32)

    def _present(self, column: str) -> pd.Series:
        """The column with an absent name marked as absent rather than as a name.

        The notes write "no site recorded" as an empty string, which ``get_dummies``
        would otherwise give a column of its own. Absence is not one of the places.
        """
        values = self._index[column].fillna("").astype(str).str.strip()
        return values.where(values != "", other=None)

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def matrix(self, rows: np.ndarray) -> RowView:
        return RowView(self._matrix, rows)

    def feature_names(self) -> list[str]:
        """One name per column, so a gain ranking says which site rather than which axis.

        This used to report ``site_code``, a single entry standing for all 47 of them,
        which meant the importance tables could say the site mattered and never which
        one. A column per name is more of them and says more.
        """
        return [name for block in self._name_columns for name in block] + self._numeric


class LogbookFeatureSource(TabularFeatureSource):
    """Everything written down about a recording, and none of the recording.

    Both the species task and the call type tasks are measured against this. There
    used to be a narrower control for call types, seeing the site, the coordinates,
    the collection and the noise conditions but not the four header fields. That was
    a mistake and an expensive one. Clip duration alone predicts most call type
    labels, because a note is written against a whole cut and a longer cut is more
    likely to contain any given call, so the control was being asked to clear a bar
    while blindfolded on the one field that mattered. Handing it back moved the only
    call type result this project reported from +0.101 at p = 0.02 to +0.005 at
    p = 0.77.

    The metadata control is the one that still misses something, deliberately: the
    gap between it and this says how much of the floor the rest of the paperwork was
    carrying.

    The collection code earns its place here on measurement. Three codes carry 97.6%
    of the clips that have one, which is 89.2% of the clips under study, each sits in
    exactly one species, and they span 61, 51 and 12 tapes, so a fold boundary drawn
    between tapes does nothing to hide them.
    """

    CATEGORICAL = ("site", "collection_code")
    NUMERIC = (
        "native_sample_rate",
        "year",
        "duration_seconds",
        "bytes_on_disk",
        "latitude",
        "longitude",
    )

    def __init__(self, index: pd.DataFrame, condition_columns: list[str]):
        super().__init__(
            index,
            name="logbook",
            categorical=list(self.CATEGORICAL),
            numeric=[*self.NUMERIC, *condition_columns],
        )


class MetadataFeatureSource(FeatureSource):
    """Recording metadata only, with no audio content whatsoever.

    This is the control. Native sample rate, recording year, clip duration and file
    size describe the tape and the equipment. None of them describe the animal.
    Whatever accuracy this reaches is the floor an audio model has to clear before its
    score can be read as evidence about whale vocalisation.
    """

    COLUMNS = ("native_sample_rate", "year", "duration_seconds", "bytes_on_disk")

    def __init__(self, index: pd.DataFrame):
        missing = set(self.COLUMNS) - set(index.columns)
        if missing:
            raise ValueError(f"window index is missing metadata columns: {sorted(missing)}")
        self._index = index.reset_index(drop=True)
        self._matrix = self._index.loc[:, list(self.COLUMNS)].to_numpy(dtype=np.float32)

    @property
    def name(self) -> str:
        return "metadata"

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def matrix(self, rows: np.ndarray) -> RowView:
        return RowView(self._matrix, rows)

    def feature_names(self) -> list[str]:
        return list(self.COLUMNS)
