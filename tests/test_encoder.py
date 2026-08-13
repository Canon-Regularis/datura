"""The frozen encoder, and the two things it must not do.

It must not touch the network when nobody asked for a window. Building an extractor
happens whenever a cache path is needed or a report asks what a model was given, and
a download on that path would make the offline jobs fail for no reason.

And it must not weaken the rule that keeps native sample rate out of the features.
The encoder upsamples, which is the one thing ``to_target_rate`` refuses to do, so
the precondition that makes it safe is asserted rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.audio.resample import UpsamplingRejected, to_encoder_rate, to_target_rate
from src.config import PROJECT_ROOT, Config, ConfigError, EncoderConfig, load_config
from src.features import registry
from src.features.encoder import ARCHITECTURES, EncoderEmbedding, EncoderUnavailable

from .conftest import write_config

TINY = {
    "architecture": "wav2vec2_tiny",
    "checkpoint": "",
    "embedding_dim": 32,
    "layer": 1,
    "batch_size": 4,
}


def tiny_config(tmp_path: Path) -> Config:
    return load_config(write_config(tmp_path, encoder=TINY))


def test_building_an_extractor_loads_no_weights(tmp_path):
    """Every offline caller goes through here, so it has to stay side effect free."""
    cfg = load_config(write_config(tmp_path, encoder={**TINY, "checkpoint": ""}))
    extractor = registry.build_extractor(registry.ENCODER, cfg)

    assert extractor.name == "encoder"
    assert extractor.output_shape(cfg.audio.window_samples) == (32,)
    assert extractor.cache_sections == ("dataset", "audio", "encoder")
    assert len(extractor.feature_names()) == 32
    # The laziness is the claim, and only the private handle can show it. A public
    # accessor for it would exist solely to be read here.
    assert extractor._module is None, "asking a shape must not build the model"  # noqa: SLF001


def test_the_encoder_cache_key_is_its_own(tmp_path):
    """Editing the encoder must not discard the spectrogram cache, or the reverse."""
    first = tiny_config(tmp_path / "a")
    second = load_config(write_config(tmp_path / "b", encoder={**TINY, "layer": 2}))

    assert first.audio_digest == second.audio_digest
    assert first.spectrogram_digest == second.spectrogram_digest
    assert first.encoder_digest != second.encoder_digest


def test_a_checkpoint_without_a_url_is_refused():
    with pytest.raises(ConfigError, match="url"):
        EncoderConfig(checkpoint="weights.pth", url="")


def test_an_unknown_pooling_is_refused():
    with pytest.raises(ConfigError, match="pooling"):
        EncoderConfig(pooling="median")


def test_a_layer_below_one_is_refused():
    with pytest.raises(ConfigError, match="layer"):
        EncoderConfig(layer=0)


def test_a_width_the_architecture_cannot_produce_is_refused(tmp_path):
    pytest.importorskip("torchaudio")
    cfg = load_config(write_config(tmp_path, encoder={**TINY, "embedding_dim": 999}))
    extractor = registry.build_extractor(registry.ENCODER, cfg)
    windows = np.zeros((1, cfg.audio.window_samples), dtype=np.float32)

    with pytest.raises(EncoderUnavailable, match="embedding_dim"):
        extractor.transform_batch(windows, cfg.audio.target_sample_rate)


def test_a_layer_deeper_than_the_stack_is_refused(tmp_path):
    pytest.importorskip("torchaudio")
    cfg = load_config(write_config(tmp_path, encoder={**TINY, "layer": 99}))
    extractor = registry.build_extractor(registry.ENCODER, cfg)
    windows = np.zeros((1, cfg.audio.window_samples), dtype=np.float32)

    with pytest.raises(EncoderUnavailable, match="layer"):
        extractor.transform_batch(windows, cfg.audio.target_sample_rate)


def test_an_untrained_encoder_still_runs_the_whole_path(tmp_path):
    """No checkpoint, no download, and the cache still fills with the declared width."""
    pytest.importorskip("torchaudio")
    cfg = tiny_config(tmp_path)
    extractor = registry.build_extractor(registry.ENCODER, cfg)

    rng = np.random.default_rng(0)
    windows = rng.standard_normal((3, cfg.audio.window_samples)).astype(np.float32)
    out = extractor.transform_batch(windows, cfg.audio.target_sample_rate)

    assert out.shape == (3, 32)
    assert out.dtype == np.float16
    assert np.isfinite(out).all()


def test_one_window_and_a_batch_of_one_agree(tmp_path):
    pytest.importorskip("torchaudio")
    cfg = tiny_config(tmp_path)
    extractor = registry.build_extractor(registry.ENCODER, cfg)

    rng = np.random.default_rng(1)
    window = rng.standard_normal(cfg.audio.window_samples).astype(np.float32)

    single = extractor.transform(window, cfg.audio.target_sample_rate)
    batched = extractor.transform_batch(window[np.newaxis, :], cfg.audio.target_sample_rate)
    assert np.array_equal(single, batched[0])


def test_audio_at_the_wrong_rate_is_refused(tmp_path):
    cfg = tiny_config(tmp_path)
    extractor = registry.build_extractor(registry.ENCODER, cfg)
    windows = np.zeros((1, cfg.audio.window_samples), dtype=np.float32)

    with pytest.raises(ValueError, match="Hz"):
        extractor.transform_batch(windows, cfg.audio.target_sample_rate * 2)


def test_the_config_layer_refuses_a_corpus_that_would_be_upsampled(tmp_path):
    """Where the precondition is actually enforced, and it predates the encoder."""
    with pytest.raises(ConfigError, match="min_native_sample_rate"):
        load_config(write_config(tmp_path, audio={"min_native_sample_rate": 4000}))


def test_the_encoder_restates_that_precondition_for_itself(tmp_path):
    """A second guard, so loosening the rule above cannot quietly reach the encoder.

    It cannot be reached through a real config, because ``AudioConfig`` refuses one
    first. A stand in carries the two fields the check reads.
    """
    cfg = tiny_config(tmp_path)
    loosened = SimpleNamespace(
        audio=SimpleNamespace(min_native_sample_rate=5120, target_sample_rate=10000),
        encoder=cfg.encoder,
    )
    with pytest.raises(EncoderUnavailable, match="shares one rate"):
        EncoderEmbedding(loosened)


def test_the_pipeline_still_refuses_to_upsample_a_native_file():
    """``to_encoder_rate`` is an exception for one call site and nowhere else."""
    signal = np.zeros(1000, dtype=np.float32)
    with pytest.raises(UpsamplingRejected):
        to_target_rate(signal, native_rate=5120, target_rate=10000)

    lifted = to_encoder_rate(signal, current_rate=10000, encoder_rate=16000)
    assert len(lifted) == pytest.approx(1600, abs=8), "the encoder path does upsample"

    unchanged = to_encoder_rate(signal, current_rate=16000, encoder_rate=16000)
    assert len(unchanged) == len(signal)


@pytest.mark.parametrize("name", ["base.yaml", "base_5k.yaml", "wide.yaml"])
def test_every_committed_config_names_real_weights(name):
    """An empty checkpoint is for the synthetic corpus, never for a published result."""
    cfg = load_config(f"configs/{name}")

    assert cfg.encoder.is_trained, f"{name} would publish embeddings from random weights"
    assert cfg.encoder.url.startswith("https://")
    assert len(cfg.encoder.sha256) == 64
    assert cfg.encoder.architecture in ARCHITECTURES


@pytest.mark.parametrize("name", ["base.yaml", "base_5k.yaml", "wide.yaml"])
def test_every_committed_config_declares_the_encoder_somewhere_in_its_chain(name):
    """Leaning on the dataclass default would hide the digest from the file that set it.

    Stated rather than defaulted, and a variant may inherit the statement: the section
    is nine lines and pinning a checkpoint in five files is how one of them ends up
    naming weights the other four have moved off.
    """
    seen = []
    path = PROJECT_ROOT / "configs" / name
    while path is not None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        seen.append(path.name)
        if "encoder" in raw:
            return
        parent = raw.get("extends")
        path = (path.parent / str(parent)).resolve() if parent else None

    pytest.fail(f"{name} defaults the encoder; nothing in {seen} declares it")


def test_torchaudio_is_the_same_build_as_torch():
    """Their native libraries have to match, so they must come from the same index.

    torchaudio's Linux wheel links against a specific libtorch. Taking it from PyPI
    while torch comes from the pytorch index installs cleanly, passes on Windows where
    the wheel carries no such extension, and then fails at import on Linux with an
    undefined symbol. That is how it reached CI as three red jobs.

    The local version tag is what tells them apart. An index build carries ``+cpu`` or
    ``+cu126``; a PyPI build carries nothing.
    """
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")

    def build(version: str) -> str:
        return version.partition("+")[2]

    assert build(torchaudio.__version__) == build(torch.__version__), (
        f"torch is {torch.__version__} and torchaudio is {torchaudio.__version__}, "
        "so they came from different indexes and their native libraries will not match"
    )
