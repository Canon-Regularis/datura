from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.config import PROJECT_ROOT, ConfigError, load_config
from tests.conftest import write_config


def tmp_model_file(payload: dict) -> Path:
    """A model hyperparameter file on disk, for the validation tests below."""
    path = Path(tempfile.mkdtemp()) / "model.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


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


def test_a_misspelled_section_is_refused_rather_than_defaulted(tmp_path):
    """The hole this closed, which was a correctness problem rather than a nicety.

    Keys inside a section were checked and the section names were not. Writing
    ``pipelines:`` left ``PipelineConfig.models`` as ``None``, which allows every
    model, so ``wide.yaml`` would have trained the two networks it exists to exclude
    and rewritten a committed report. Writing ``encodder:`` left the checkpoint empty
    and the encoder ran with randomly initialised weights behind a log warning.
    """
    from src.config.loading import SECTIONS

    source = PROJECT_ROOT / "configs" / "base.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["pipeline"] = {"models": ["xgboost"]}

    for real, typo in (("pipeline", "pipelines"), ("encoder", "encodder"), ("split", "splitt")):
        body = dict(raw)
        assert real in body, f"{source.name} no longer carries {real}"
        assert real in SECTIONS
        body[typo] = body.pop(real)

        path = tmp_path / f"{typo}.yaml"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")
        with pytest.raises(ConfigError, match=typo):
            load_config(path)


def test_every_committed_config_uses_only_known_sections():
    """And the guard above does not refuse anything the project actually ships."""
    for path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        # A variant states only its differences, so what marks an experiment config is
        # the corpus it describes or the one it inherits, rather than any one section.
        if "dataset" not in raw and "extends" not in raw:
            continue  # a model hyperparameter file, not an experiment
        load_config(path)


def test_a_typo_in_a_model_hyperparameter_file_is_refused():
    """Nothing checked these files, and every setting is read with a default.

    A misspelled key fell back to a value the file did not contain and the run carried
    on without a word. The four defaults behind ``configs/cnn.yaml`` all disagree with
    it: the file says 30 epochs and the fallback is 40, 64 against 32, 0.004 against
    0.003, 7 against 8. Misspelling ``epochs`` trained a third longer than the file said.
    """
    import yaml as yaml_module

    from src.models import registry as models

    spec = models.get("cnn")
    raw = yaml_module.safe_load((PROJECT_ROOT / spec.config_file).read_text(encoding="utf-8"))

    for block, real, typo in (
        ("train", "epochs", "epocs"),
        ("augment", "noise_std", "noise_stdev"),
    ):
        broken = {name: dict(values) for name, values in raw.items()}
        broken[block][typo] = broken[block].pop(real)
        raise_on = tmp_model_file(broken)
        with pytest.raises(ConfigError, match=typo):
            models.load_settings(replace(spec, config_file=str(raise_on)))


def test_a_stray_block_in_a_model_file_is_refused():
    import yaml as yaml_module

    from src.models import registry as models

    spec = models.get("cnn")
    raw = yaml_module.safe_load((PROJECT_ROOT / spec.config_file).read_text(encoding="utf-8"))
    raw["trian"] = raw.pop("train")

    with pytest.raises(ConfigError, match="trian"):
        models.load_settings(replace(spec, config_file=str(tmp_model_file(raw))))


def test_every_committed_model_file_supplies_every_setting_its_code_reads():
    """So the defaults behind them are never the value that gets used.

    Validation stops a typo. This stops the quieter version, where a key is simply
    absent and the fallback silently becomes the setting. The mapping below refuses to
    hand over a default, so reading one raises rather than passing.
    """
    from src.models import registry as models

    class NoDefaults(dict):
        def get(self, key, default=None):
            if key not in self:
                raise AssertionError(f"{key} was read from a default rather than from the file")
            return self[key]

    for spec in models.specs():
        settings = models.load_settings(spec)
        strict = {block: NoDefaults(values) for block, values in settings.items()}
        for block, allowed in spec.settings_schema.items():
            if allowed is None or block not in strict:
                continue
            missing = allowed - set(strict[block])
            assert not missing, (
                f"{spec.config_file} leaves {sorted(missing)} in {block} to a default"
            )


def test_a_variant_inherits_what_it_does_not_state(tmp_path):
    """The whole point of ``extends``, and what it is allowed to leave out.

    ``base_5k.yaml`` names five settings. Everything else it runs on, including the
    archive checksum and the nine line encoder block, comes from ``base.yaml``, which
    is the only place either is now written down.
    """
    base = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    variant = load_config(PROJECT_ROOT / "configs" / "base_5k.yaml")

    assert variant.audio.target_sample_rate == 5120, "what it states"
    assert variant.spectrogram.fmax == 2500
    assert variant.name == "base_5k"

    assert variant.dataset.archive_sha256 == base.dataset.archive_sha256, "what it inherits"
    assert variant.encoder == base.encoder
    assert variant.split == base.split
    assert variant.audio.window_seconds == base.audio.window_seconds
    assert variant.spectrogram.n_mels == base.spectrogram.n_mels


def test_a_section_merges_key_by_key_and_a_list_replaces(tmp_path):
    """Two different rules, and confusing them would be quiet either way.

    Merging keys is what lets a variant change one band setting and inherit the other
    six. Replacing lists is what lets ``wide.yaml`` name eleven species and get eleven
    rather than fourteen, and what lets a narrower model roster actually be narrower.
    """
    wide = load_config(PROJECT_ROOT / "configs" / "wide.yaml")
    base = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    assert len(wide.dataset.species) == 11
    assert wide.dataset.zip_name == base.dataset.zip_name, "the rest of the section survives"

    narrow = load_config(PROJECT_ROOT / "configs" / "context.yaml")
    assert narrow.pipeline.models is not None
    assert "cnn" not in narrow.pipeline.models, "a roster replaces rather than extends"


def test_a_config_that_extends_itself_is_refused(tmp_path):
    """A cycle would otherwise recurse until the interpreter gave up."""
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\nname: first\n", encoding="utf-8")
    second.write_text("extends: first.yaml\nname: second\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="extends itself"):
        load_config(first)


def test_extending_a_file_that_is_not_there_says_so(tmp_path):
    path = tmp_path / "orphan.yaml"
    path.write_text("extends: nowhere.yaml\nname: orphan\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="not found"):
        load_config(path)


def test_the_variants_carry_no_setting_they_did_not_change():
    """What the collapse is for: one place to correct the checksum, not five.

    A variant that restates an inherited value is not wrong, and it is the thing that
    drifts. This keeps the four experiment variants stating only their differences.
    """
    base = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))

    for name in ("base_5k", "wide", "context", "context_shuffled"):
        raw = yaml.safe_load((PROJECT_ROOT / f"configs/{name}.yaml").read_text(encoding="utf-8"))
        assert raw.get("extends") == "base.yaml", f"{name} should inherit rather than restate"
        for section, block in raw.items():
            if not isinstance(block, dict):
                continue
            repeated = {k: v for k, v in block.items() if base.get(section, {}).get(k) == v}
            assert not repeated, f"{name}.{section} restates {sorted(repeated)} from base.yaml"
