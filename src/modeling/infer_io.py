"""Load checkpoint, build model, run chip inference (optional MC dropout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.modeling.infer_mc import mc_dropout_predict
from src.modeling.model import TomatoUNet


def build_model_from_cfg(cfg: dict[str, Any], device: torch.device) -> TomatoUNet:
    mcfg = cfg.get("model", {})
    in_ch = int(mcfg.get("in_channels", 64))
    base = int(mcfg.get("base_channels", 32))
    dropout_p = float(mcfg.get("dropout_p", 0.1))
    model = TomatoUNet(in_channels=in_ch, base=base, dropout_p=dropout_p).to(device)
    return model


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("cfg") or {}
    model = build_model_from_cfg(cfg, device)
    model.load_state_dict(ckpt["model"])
    return model, cfg


@torch.no_grad()
def predict_chip_deterministic(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    logits = model(x.to(device))
    return torch.sigmoid(logits)


def predict_chip(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    *,
    mc_samples: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Returns ``mean_prob`` (1,1,H,W), ``var_prob`` same shape if ``mc_samples > 0`` else None."""
    if mc_samples <= 0:
        p = predict_chip_deterministic(model, x, device)
        return p, None
    mean_p, var_p = mc_dropout_predict(model, x, n_samples=mc_samples, device=device)
    return mean_p, var_p
