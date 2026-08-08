from __future__ import annotations

import pytest

from src.config import ConfigError, load_config
from tests.conftest import write_config


def test_valid_config_exposes_derived_sizes(config):
    assert config.audio.window_samples == 20_000
    assert config.audio.hop_samples == 10_000
    assert config.audio.nyquist == 5_000
    assert config.dataset.label_to_index["HumpbackWhale"] == 0


def test_rejects_upsampling_configurations(tmp_path):
    path = write_config(tmp_path, audio={"min_native_sample_rate": 5120})
    with pytest.raises(ConfigError, match="upsampled"):
        load_config(path)


def test_rejects_a_band_above_nyquist(tmp_path):
    path = write_config(tmp_path, spectrogram={"fmax": 8000})
    with pytest.raises(ConfigError, match="Nyquist"):
        load_config(path)


def test_rejects_unknown_keys(tmp_path):
    path = write_config(tmp_path, split={"unexpected": 1})
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(path)


def test_rejects_missing_keys(tmp_path):
    path = write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace("  tape_id_length: 5\n", "")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(path)


def test_digest_changes_when_a_relevant_setting_changes(tmp_path):
    base = load_config(write_config(tmp_path / "a"))
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    other = load_config(
        write_config(tmp_path / "b", audio={"target_sample_rate": 8000}, spectrogram={"fmax": 3900})
    )

    assert base.audio_digest != other.audio_digest
    assert base.spectrogram_digest != other.spectrogram_digest


def test_digest_is_stable_across_loads(tmp_path):
    path = write_config(tmp_path)
    assert load_config(path).spectrogram_digest == load_config(path).spectrogram_digest


def test_no_committed_model_config_lets_the_machine_choose_its_thread_count():
    """``n_jobs: -1`` makes a committed score depend on the cores that wrote it.

    XGBoost sums its histograms in whatever order the threads finish, so the fitted
    model is not thread count invariant. Every published tree number here was fitted
    with this pinned, and a rerun on a machine with a different core count has to
    reproduce it.
    """
    import yaml

    from src.config import PROJECT_ROOT

    offenders = []
    for path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        settings = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        model = settings.get("model")
        if isinstance(model, dict) and model.get("n_jobs") == -1:
            offenders.append(path.name)

    assert not offenders, f"{offenders} would let the machine pick its own thread count"
