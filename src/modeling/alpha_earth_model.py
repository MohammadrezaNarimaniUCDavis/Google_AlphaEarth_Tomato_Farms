"""Simple baseline model for AlphaEarth tomato vs non_tomato.

Architecture:
- Encoder: small convolutional stack over 64-band input.
- Pixel head: per-pixel tomato probability (for maps + weak supervision).
- Chip head: global average pooled features → tomato/non_tomato logit.

This is a starting point; you can later swap in a deeper U-Net-style model.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(x)


class AlphaEarthTomatoModel(nn.Module):
    """Baseline multi-task model.

    Inputs:
        x: (B, C=64, H, W)

    Outputs:
        pixel_logits: (B, 1, H, W)   – per-pixel tomato logit
        chip_logits:  (B,)           – per-chip tomato logit
    """

    def __init__(self, in_channels: int = 64, base_channels: int = 32, dropout_p: float = 0.1) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels, dropout_p=dropout_p),
            nn.MaxPool2d(2),
            ConvBlock(base_channels, base_channels * 2, dropout_p=dropout_p),
        )

        # Pixel head: simple upsampling back to original size.
        self.pixel_head = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

        # Chip head: global average pooling over encoder features.
        self.chip_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(base_channels, 1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        feats = self.encoder(x)
        pixel_logits = self.pixel_head(feats)  # (B,1,H,W)
        chip_logits = self.chip_head(feats).squeeze(-1)  # (B,)
        return pixel_logits, chip_logits

