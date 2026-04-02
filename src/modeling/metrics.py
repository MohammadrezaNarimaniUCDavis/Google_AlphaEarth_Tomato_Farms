"""Pixel metrics on valid mask."""

from __future__ import annotations

import torch


@torch.no_grad()
def binary_confusion_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Micro-level counts over masked pixels (sums over batch and spatial dims)."""
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
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(
    tp: float,
    fp: float,
    fn: float,
    tn: float,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Accuracy, precision, recall, F1, IoU from aggregated confusion counts."""
    total = tp + tn + fp + fn
    acc = (tp + tn) / (total + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "iou": float(iou),
    }


@torch.no_grad()
def pixel_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Returns accuracy, precision, recall, F1, IoU on masked pixels (one batch or full tensor)."""
    c = binary_confusion_counts(logits, target, mask, threshold=threshold)
    out = metrics_from_counts(
        float(c["tp"].item()),
        float(c["fp"].item()),
        float(c["fn"].item()),
        float(c["tn"].item()),
    )
    return out


@torch.no_grad()
def chip_level_correct_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[int, int]:
    """Per-chip majority rule: mean p vs mean label. Returns (n_correct, n_chips_used)."""
    prob = torch.sigmoid(logits)
    m = mask.clamp(0, 1)
    b = logits.shape[0]
    correct = 0
    used = 0
    for i in range(b):
        mb = m[i, 0]
        denom = mb.sum().clamp(min=0.0)
        if float(denom.item()) < 1e-6:
            continue
        mean_p = (prob[i, 0] * mb).sum() / denom
        mean_y = (target[i, 0] * mb).sum() / denom
        pred = mean_p >= threshold
        lbl = mean_y >= 0.5
        if bool(pred.item()) == bool(lbl.item()):
            correct += 1
        used += 1
    return correct, used


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
