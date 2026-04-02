"""MC-Dropout: mean probability and variance per pixel (optional script use)."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    *,
    n_samples: int = 20,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns mean probability and variance of probability across forward passes."""
    device = device or x.device
    model.train()  # dropout ON
    probs = []
    for _ in range(n_samples):
        logits = model(x.to(device))
        p = torch.sigmoid(logits)
        probs.append(p)
    stacked = torch.stack(probs, dim=0)
    mean_p = stacked.mean(dim=0)
    var_p = stacked.var(dim=0, unbiased=False)
    return mean_p, var_p
