"""Augmentation, applied in mel space.

Gain jitter is absent by design: windows are normalised individually during
extraction, so scaling one would be undone before the model ever saw it. Pitch
shifting is absent because it moves the frequency content the label depends on.
"""

from __future__ import annotations

from typing import Any

import torch


class SpectrogramAugment:
    """Masking and shifting applied in mel space.

    Gain jitter is absent by design. Windows are normalised individually during
    extraction, so scaling one would be undone before the model ever sees it.
    Pitch shifting is absent because it moves the frequency content the label
    depends on.
    """

    def __init__(self, settings: dict[str, Any]):
        self.enabled = bool(settings.get("enabled", True))
        self.probability = float(settings.get("probability", 0.5))
        self.max_time_shift = float(settings.get("max_time_shift", 0.2))
        self.noise_std = float(settings.get("noise_std", 0.1))
        self.freq_mask_bins = int(settings.get("freq_mask_bins", 8))
        self.time_mask_frames = int(settings.get("time_mask_frames", 32))

    def __call__(self, batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if not self.enabled:
            return batch
        n, _, n_mels, n_frames = batch.shape
        device = batch.device

        def coin() -> torch.Tensor:
            return torch.rand(n, device=device, generator=generator) < self.probability

        if self.max_time_shift > 0:
            span = max(int(self.max_time_shift * n_frames), 1)
            shifts = torch.randint(-span, span + 1, (n,), device=device, generator=generator)
            shifts = torch.where(coin(), shifts, torch.zeros_like(shifts))
            for i, shift in enumerate(shifts.tolist()):
                if shift:
                    batch[i] = torch.roll(batch[i], shifts=shift, dims=-1)

        if self.noise_std > 0:
            noise = torch.randn(batch.shape, device=device, generator=generator) * self.noise_std
            batch = batch + noise * coin().view(n, 1, 1, 1)

        batch = self._mask(batch, coin(), self.freq_mask_bins, n_mels, 2, generator)
        return self._mask(batch, coin(), self.time_mask_frames, n_frames, 3, generator)

    @staticmethod
    def _mask(
        batch: torch.Tensor,
        selected: torch.Tensor,
        max_width: int,
        axis_size: int,
        axis: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if max_width <= 0:
            return batch
        width = int(min(max_width, axis_size - 1))
        positions = torch.arange(axis_size, device=batch.device)
        for i in torch.nonzero(selected).flatten().tolist():
            size = int(torch.randint(1, width + 1, (1,), generator=generator, device=batch.device))
            start = int(
                torch.randint(
                    0, axis_size - size + 1, (1,), generator=generator, device=batch.device
                )
            )
            span = (positions >= start) & (positions < start + size)
            shape = [1, 1, 1, 1]
            shape[axis] = axis_size
            batch[i] = batch[i].masked_fill(span.view(shape[1:]), 0.0)
        return batch
