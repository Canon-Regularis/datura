"""Grad-CAM over log mel windows.

Answers where in time and frequency the CNN's evidence sits. Treat it as a lead
rather than a measurement: it shows what the last convolutional stage responded to,
not what the model would lose if that region were absent. ``src.evaluate.occlusion``
answers the second question and the two belong together.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from src.models.cnn import SpectrogramCNN


class GradCam:
    """Class activation maps from the final residual stage."""

    def __init__(self, model: SpectrogramCNN):
        if model.module is None:
            raise RuntimeError("the model must be fitted before Grad-CAM can run")
        self.model = model
        self.module = model.module
        self.device = model.device

    def heatmaps(
        self, windows: np.ndarray, targets: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(heatmaps, predicted_classes)`` for a batch of windows.

        Heatmaps are resized to the input grid and scaled to [0, 1] per window.
        """
        self.module.eval()
        inputs = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)

        captured: dict[str, torch.Tensor] = {}

        def capture(_module, _inputs, output: torch.Tensor) -> None:
            captured["activations"] = output

        handle = self.module.final_stage.register_forward_hook(capture)  # type: ignore[union-attr]
        try:
            logits = self.module(inputs)
            predicted = logits.argmax(dim=1)
            chosen = (
                predicted
                if targets is None
                else torch.as_tensor(targets, dtype=torch.long, device=self.device)
            )
            selected = logits.gather(1, chosen.view(-1, 1)).sum()
            gradients = torch.autograd.grad(selected, captured["activations"])[0]
        finally:
            handle.remove()

        activations = captured["activations"].detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        maps = F.relu((weights * activations).sum(dim=1, keepdim=True))
        maps = F.interpolate(maps, size=inputs.shape[-2:], mode="bilinear", align_corners=False)

        flat = maps.flatten(1)
        peak = flat.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        normalised = (flat / peak).view(maps.shape[0], *inputs.shape[-2:])
        return normalised.cpu().numpy(), predicted.cpu().numpy()
