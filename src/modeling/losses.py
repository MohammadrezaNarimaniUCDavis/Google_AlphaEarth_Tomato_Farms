"""Masked BCE + soft Dice on valid pixels."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """logits, target, mask: (N,1,H,W). Loss averaged over masked pixels."""
    m = mask.clamp(0, 1)
    if m.sum() < 1:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    loss = (loss * m).sum() / m.sum().clamp(min=1.0)
    return loss


def soft_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """1 - Dice for binary target in [0,1]."""
    p = torch.sigmoid(logits)
    m = mask.clamp(0, 1)
    t = target * m
    p = p * m
    inter = (p * t).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps
    dice = (2 * inter + eps) / denom
    return 1.0 - dice.mean()


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = masked_bce_with_logits(logits, target, mask, pos_weight=pos_weight)
    dice = soft_dice(logits, target, mask)
    return bce_weight * bce + dice_weight * dice
