"""The architecture.

A stem, a few residual stages, global pooling and a linear head. Small on purpose:
the species set survives on a few dozen independent tapes, so capacity is not the
limiting factor, and a larger network only memorises tapes faster.
"""

from __future__ import annotations

import torch
from torch import nn


class _BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class MelResNet(nn.Module):
    """Stem, four residual stages, global pooling, linear head."""

    def __init__(
        self,
        n_classes: int,
        base_width: int = 32,
        n_stages: int = 4,
        blocks_per_stage: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        in_channels = base_width
        for stage in range(n_stages):
            out_channels = base_width * (2**stage)
            for block in range(blocks_per_stage):
                stride = 2 if (block == 0 and stage > 0) else 1
                stages.append(_BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, n_classes)

    @property
    def final_stage(self) -> nn.Module:
        """The block Grad-CAM hooks into."""
        return self.stages[-1]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(pooled))
