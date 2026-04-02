"""Aggregate per-pixel predictions to chip / polygon summaries."""

from __future__ import annotations

import torch


def chip_aggregate(
    prob: torch.Tensor,
    mask: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    *,
    unc_high_q: float = 0.75,
) -> dict[str, float]:
    """``prob``, ``mask``: (1,1,H,W) or (1,H,W); optional ``uncertainty`` same shape (e.g. MC variance).

    Returns mean/median tomato probability over valid pixels, optional uncertainty summaries.
    """
    p = prob.squeeze()
    m = mask.squeeze().clamp(0, 1)
    valid = m > 0.5
    if valid.sum() < 1:
        return {
            "mean_p_tomato": float("nan"),
            "median_p_tomato": float("nan"),
            "frac_valid": 0.0,
            "mean_uncertainty": float("nan"),
            "frac_high_uncertainty": float("nan"),
        }
    pv = p[valid]
    out: dict[str, float] = {
        "mean_p_tomato": float(pv.mean().item()),
        "median_p_tomato": float(pv.median().item()),
        "frac_valid": float(valid.float().mean().item()),
    }
    if uncertainty is not None:
        u = uncertainty.squeeze()[valid]
        thresh = torch.quantile(u.float(), unc_high_q)
        out["mean_uncertainty"] = float(u.mean().item())
        out["frac_high_uncertainty"] = float((u >= thresh).float().mean().item())
    else:
        out["mean_uncertainty"] = float("nan")
        out["frac_high_uncertainty"] = float("nan")
    return out
