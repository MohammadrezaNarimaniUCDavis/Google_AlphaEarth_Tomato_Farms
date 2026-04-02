"""Pixel metrics on valid mask."""

from __future__ import annotations

import torch


@torch.no_grad()
def pixel_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Returns accuracy, precision, recall, IoU on masked pixels."""
    m = mask.clamp(0, 1) > 0.5
    prob = torch.sigmoid(logits)
    pred = prob >= threshold
    tgt = target >= 0.5
    pred = pred & m
    tgt = tgt & m

    tp = (pred & tgt).sum().float()
    fp = (pred & ~tgt).sum().float()
    fn = (~pred & tgt).sum().float()
    tn = (~pred & ~tgt).sum().float()

    total = m.sum().float().clamp(min=1.0)
    acc = (tp + tn) / total
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "acc": float(acc.item()),
        "precision": float(prec.item()),
        "recall": float(rec.item()),
        "iou": float(iou.item()),
    }


@torch.no_grad()
def chip_level_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Mean probability over chip vs chip label (0/1) — quick scalar."""
    prob = torch.sigmoid(logits)
    m = mask.clamp(0, 1)
    mean_p = (prob * m).sum() / m.sum().clamp(min=1.0)
    label = (target * m).sum() / m.sum().clamp(min=1.0)
    cls = (mean_p >= threshold).float()
    return float((cls == (label >= 0.5).float()).item())
