"""A frozen pretrained encoder, turning a window into one vector.

Every other representation here is computed from the recording by code in this
repository. This one is a set of weights trained on somebody else's audio, and it is
in the project to answer the obvious objection to its headline: that a model given no
audio beats the audio models because the audio models are weak.

The encoder is never fitted. Windows go through it once, the embeddings are cached
like any other representation, and only a small head is trained per fold. That keeps
the comparison honest, since the probe and the networks then see the same folds, and
it keeps the cost down, because a fifty split run over cached vectors is minutes.

Two decisions worth knowing about.

The weights load lazily. Constructing this class touches neither the network nor the
disk, so asking a cache whether it exists, or asking the explainer for a frequency
axis, never triggers a download.

The window is upsampled to meet the encoder. This is the only upsampling in the
project, and ``src.audio.resample.to_encoder_rate`` carries the argument for why it
does not reintroduce the leak the rest of the pipeline is built to avoid. The short
version: every window reaching here is already at one common rate, so the empty band
above the old Nyquist is identical for every species and separates nothing. That
precondition is asserted rather than assumed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.audio.resample import to_encoder_rate
from src.config import Config
from src.errors import DaturaError
from src.features.base import FeatureExtractor

logger = logging.getLogger(__name__)

# Architectures this project knows how to build, and the keyword arguments that
# differ from the torchaudio defaults. The tiny one exists so the test suite can
# exercise the whole extraction path offline, with no weights and no download.
ARCHITECTURES: dict[str, dict[str, Any]] = {
    "wav2vec2_base": {
        "encoder_embed_dim": 768,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_ff_interm_features": 3072,
    },
    "wav2vec2_tiny": {
        "encoder_embed_dim": 32,
        "encoder_num_layers": 2,
        "encoder_num_heads": 2,
        "encoder_ff_interm_features": 64,
    },
}


class EncoderUnavailable(DaturaError):
    """Raised when the encoder cannot be built or its weights cannot be found."""


def _build_module(cfg: Config):
    """The torchaudio model, on the device that will run it.

    Imported inside the call so that reading the extractor roster, or checking
    whether a cache exists, does not pull torch into the process.
    """
    try:
        import torch
        from torchaudio.models import wav2vec2_model
    except ImportError as error:  # pragma: no cover - exercised by the install, not the suite
        raise EncoderUnavailable(
            "the encoder needs torchaudio; install with uv sync --extra cpu"
        ) from error

    settings = cfg.encoder
    if settings.architecture not in ARCHITECTURES:
        raise EncoderUnavailable(
            f"unknown encoder.architecture {settings.architecture!r}; "
            f"this project builds {sorted(ARCHITECTURES)}"
        )

    shape = ARCHITECTURES[settings.architecture]
    if shape["encoder_embed_dim"] != settings.embedding_dim:
        raise EncoderUnavailable(
            f"encoder.embedding_dim is {settings.embedding_dim}, but "
            f"{settings.architecture} produces {shape['encoder_embed_dim']}"
        )
    if settings.layer > shape["encoder_num_layers"]:
        raise EncoderUnavailable(
            f"encoder.layer is {settings.layer}, but {settings.architecture} has "
            f"{shape['encoder_num_layers']} layers"
        )

    module = wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.0,
        encoder_layer_norm_first=False,
        encoder_layer_drop=0.0,
        aux_num_out=None,
        **shape,
    )

    if settings.is_trained:
        state = _weights(cfg)
        module.load_state_dict(state)
        logger.info("encoder %s loaded from %s", settings.architecture, settings.checkpoint)
    else:
        logger.warning(
            "encoder.checkpoint is empty, so %s is randomly initialised and its "
            "embeddings mean nothing; this is for tests only",
            settings.architecture,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return module.eval().to(device), device


def _weights(cfg: Config):
    """The checkpoint, fetched once and verified against the digest in the config."""
    import torch

    from src.data.download import verify_digest
    from src.data.remote import download

    settings = cfg.encoder
    destination = cfg.paths.raw / "encoders" / settings.checkpoint
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("fetching encoder weights from %s", settings.url)
        download(settings.url, destination)

    if settings.sha256:
        verify_digest(destination, settings.sha256, what="encoder checkpoint")
    return torch.load(destination, map_location="cpu", weights_only=True)


class EncoderEmbedding(FeatureExtractor):
    """One pooled vector per window, from a frozen pretrained encoder."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._settings = cfg.encoder
        self._module = None
        self._device = None

        if cfg.audio.min_native_sample_rate < cfg.audio.target_sample_rate:
            raise EncoderUnavailable(
                "the encoder upsamples, which is only safe because every kept clip "
                "shares one rate; this config allows a clip below the target rate"
            )

    @property
    def name(self) -> str:
        return "encoder"

    @property
    def cache_sections(self) -> tuple[str, ...]:
        return ("dataset", "audio", "encoder")

    @property
    def storage_dtype(self) -> np.dtype:
        return np.dtype(np.float16)

    def output_shape(self, window_samples: int) -> tuple[int, ...]:
        """Declared rather than measured, so no weights load to answer a shape."""
        return (self._settings.embedding_dim,)

    def feature_names(self) -> list[str]:
        return [f"e{i}" for i in range(self._settings.embedding_dim)]

    def transform(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.transform_batch(window[np.newaxis, :], sample_rate)[0]

    def transform_batch(self, windows: np.ndarray, sample_rate: int) -> np.ndarray:
        if windows.ndim != 2:
            raise ValueError(f"expected a 2-D window array, got shape {windows.shape}")
        if sample_rate != self._cfg.audio.target_sample_rate:
            raise ValueError(
                f"encoder features expect audio at {self._cfg.audio.target_sample_rate} Hz, "
                f"got {sample_rate} Hz"
            )

        import torch

        if self._module is None:
            self._module, self._device = _build_module(self._cfg)

        resampled = np.stack(
            [to_encoder_rate(w, sample_rate, self._settings.sample_rate) for w in windows]
        )

        pooled = []
        with torch.inference_mode():
            for start in range(0, len(resampled), self._settings.batch_size):
                block = resampled[start : start + self._settings.batch_size]
                tensor = torch.from_numpy(block).to(self._device)
                layers, _ = self._module.extract_features(tensor, num_layers=self._settings.layer)
                pooled.append(self._pool(layers[-1]).cpu().numpy())

        stacked = np.concatenate(pooled)
        if stacked.shape[1] != self._settings.embedding_dim:
            raise EncoderUnavailable(
                f"encoder.embedding_dim is {self._settings.embedding_dim}, but the model "
                f"produced {stacked.shape[1]}"
            )
        return stacked.astype(self.storage_dtype, copy=False)

    def _pool(self, states):
        """Across time, leaving one vector per window."""
        if self._settings.pooling == "max":
            return states.max(dim=1).values
        return states.mean(dim=1)
