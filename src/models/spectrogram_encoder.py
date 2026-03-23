"""CNN encoder for mel spectrogram inputs."""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectrogramEncoder(nn.Module):
    """Lightweight 2D CNN that encodes (1, n_mels, time_frames) -> (embed_dim,)."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        channels = [1, 32, 64, 128, 256]
        layers: list[nn.Module] = []
        for i in range(4):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.GELU(),
                nn.MaxPool2d(2),
            ])
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, n_mels, time_frames) -> (batch, embed_dim)"""
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.proj(x)
