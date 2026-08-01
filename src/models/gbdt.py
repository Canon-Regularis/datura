"""Gradient boosted trees over hand engineered features.

Also serves the metadata control, which is the same estimator pointed at recording
metadata instead of audio. Using one estimator for both means any gap between them
comes from the features rather than from the learner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features.source import RowView
from src.models.base import Batch, FoldContext, WindowClassifier, balanced_class_weights

# The tail of a gain ranking is noise; only the top of it is worth reporting.
_IMPORTANCE_ROWS = 60


class GradientBoostedTrees(WindowClassifier):
    def __init__(self, params: dict[str, Any], name: str = "xgboost"):
        self._params = dict(params)
        self._name = name
        self._model: XGBClassifier | None = None

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _flatten(features: RowView | np.ndarray) -> np.ndarray:
        array = features.to_numpy() if isinstance(features, RowView) else np.asarray(features)
        return array.reshape(len(array), -1).astype(np.float32, copy=False)

    def fit(self, train: Batch, validation: Batch, n_classes: int) -> None:
        weights = balanced_class_weights(train.labels, n_classes)
        self._model = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            **self._params,
        )
        self._model.fit(
            self._flatten(train.features),
            train.labels,
            sample_weight=weights[train.labels],
            eval_set=[(self._flatten(validation.features), validation.labels)],
            verbose=False,
        )

    def predict_proba(self, features: RowView | np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict_proba")
        return self._model.predict_proba(self._flatten(features)).astype(np.float64)

    def feature_importance(self, feature_names: list[str] | None) -> dict[str, float]:
        """Gain based importance, keyed by feature name where names are available."""
        if self._model is None:
            raise RuntimeError("fit must be called before feature_importance")
        scores = self._model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(scores))]
        return dict(zip(names, (float(s) for s in scores), strict=True))

    def artifacts(self, context: FoldContext) -> dict[str, pd.DataFrame]:
        scores = self.feature_importance(context.feature_names)
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        table = pd.DataFrame(ranked, columns=["feature", "gain"])
        return {"feature_importance": table.head(_IMPORTANCE_ROWS)}

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("fit must be called before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path.with_suffix(".json")))
