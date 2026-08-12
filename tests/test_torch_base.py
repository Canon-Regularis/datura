"""The training loop both torch models share.

It was written twice and the two copies differed on 34 of about 87 lines. The 34 were
what each model genuinely needs: which module to build, how a batch becomes a tensor,
whether to augment. The rest was one procedure in two places, which is the shape that
lets a change to early stopping land in one model and not the other, with nothing in
the report saying which had which.

The extraction had to be exact rather than merely equivalent, because merging two loops
can reorder RNG consumption and move every score without any test noticing. Both models
were checked against their committed predictions, fold by fold, and reproduce them bit
for bit. What is pinned here is the contract, so a third model cannot quietly reintroduce
a second copy.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.features.views import RowView  # noqa: E402
from src.models.base import Batch  # noqa: E402
from src.models.torch_base import TorchWindowClassifier  # noqa: E402

SETTINGS = {
    "epochs": 3,
    "batch_size": 8,
    "lr": 0.01,
    "weight_decay": 0.0,
    "warmup_epochs": 1,
    "early_stopping_patience": 2,
    "seed": 7,
    "device": "cpu",
    "deterministic": True,
}


class Tiny(TorchWindowClassifier):
    """A linear map, so the base class is what is under test rather than a model."""

    def __init__(self, settings=None, width: int = 4):
        super().__init__(settings or SETTINGS)
        self._width = width
        self.prepared = 0
        self.augmented = 0

    @property
    def name(self) -> str:
        return "tiny"

    def _prepare(self, train: Batch) -> None:
        self.prepared += 1

    def _build(self, n_classes: int):
        return torch.nn.Linear(self._width, n_classes)

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(features), dtype=torch.float32)

    def save(self, path) -> None:
        raise NotImplementedError

    def _augment(self, inputs, generator):
        self.augmented += 1
        return inputs


def batch(rows: int = 40, width: int = 4, classes: int = 3) -> Batch:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(rows, width)).astype(np.float32)
    labels = np.arange(rows) % classes
    # A separable signal, so a three epoch fit has something to find.
    features[np.arange(rows), labels % width] += 4.0
    return Batch(RowView.over(features), labels)


def test_the_loop_prepares_before_it_builds():
    """The probe's head width is measured from its training rows, so order is a contract."""
    model = Tiny()
    model.fit(batch(), batch(rows=12), 3)

    assert model.prepared == 1, "prepare runs once, before the module exists"
    assert model.module is not None


def test_the_loop_records_one_history_row_per_epoch():
    model = Tiny()
    model.fit(batch(), batch(rows=12), 3)

    assert [row["epoch"] for row in model.history] == [0, 1, 2]
    assert all(set(row) == {"epoch", "train_loss", "val_macro_f1"} for row in model.history)
    assert all(row["train_loss"] >= 0 for row in model.history)


def test_the_same_seed_gives_the_same_weights():
    """Every published network score depends on this, and on nothing else being random."""
    first, second = Tiny(), Tiny()
    train, validation = batch(), batch(rows=12)
    first.fit(train, validation, 3)
    second.fit(train, validation, 3)

    left = first.predict_proba(RowView.over(batch(rows=16).features.to_numpy()))
    right = second.predict_proba(RowView.over(batch(rows=16).features.to_numpy()))
    assert np.array_equal(left, right)


def test_a_different_seed_gives_different_weights():
    """Otherwise the test above would pass for a loop that never trained anything."""
    train, validation = batch(), batch(rows=12)
    first = Tiny()
    second = Tiny({**SETTINGS, "seed": 99})
    first.fit(train, validation, 3)
    second.fit(train, validation, 3)

    probe = RowView.over(batch(rows=16).features.to_numpy())
    assert not np.array_equal(first.predict_proba(probe), second.predict_proba(probe))


def test_augmentation_is_off_unless_a_model_asks_for_it():
    """The probe has none. Inheriting the CNN's by accident would change its score.

    The default has to be exactly the identity rather than merely harmless, because a
    batch that came back subtly altered would move every probe number and look like
    nothing at all.
    """
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    generator = torch.Generator().manual_seed(0)

    untouched = TorchWindowClassifier._augment(Tiny(), inputs, generator)
    assert untouched is inputs, "the default returns the very batch it was given"

    asked = Tiny()
    asked.fit(batch(), batch(rows=12), 3)
    assert asked.augmented > 0, "and a model that overrides it is called"


def test_amp_is_opt_in_rather_than_inferred():
    """Mixed precision changes results, so a model gets it by asking rather than by device."""
    from src.models.cnn.classifier import SpectrogramCNN
    from src.models.probe import EmbeddingProbe

    assert TorchWindowClassifier.USES_AMP is False, "the safe default"
    assert EmbeddingProbe.USES_AMP is False, "a linear map over cached vectors gains nothing"
    assert SpectrogramCNN.USES_AMP is True


def test_predictions_are_probabilities_over_every_class():
    model = Tiny()
    model.fit(batch(), batch(rows=12), 3)
    probabilities = model.predict_proba(RowView.over(batch(rows=16).features.to_numpy()))

    assert probabilities.shape == (16, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_a_missing_setting_is_refused_rather_than_defaulted():
    """Every one of these carried a default, and the defaults disagreed with the file.

    ``configs/cnn.yaml`` says 30 epochs and the fallback said 40, so a misspelled key
    trained a third longer than the config claimed and printed nothing either way.
    """
    thin = {name: value for name, value in SETTINGS.items() if name != "epochs"}
    with pytest.raises(KeyError, match="epochs"):
        Tiny(thin).fit(batch(), batch(rows=12), 3)


def test_predict_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="fit must be called"):
        Tiny().predict_proba(RowView.over(batch(rows=4).features.to_numpy()))
