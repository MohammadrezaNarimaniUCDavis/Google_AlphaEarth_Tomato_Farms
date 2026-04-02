"""Shared one-chip inference (used by infer_chip and infer_batch CLIs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.modeling.dataset import load_chip_for_model
from src.modeling.infer_io import predict_chip


def tensors_from_row(
    row: pd.Series,
    target_hw: tuple[int, int],
    repo_root: Path,
) -> tuple[Path, str | None, dict[str, torch.Tensor], str]:
    lp = Path(row["local_path"])
    if not lp.is_absolute():
        lp = repo_root / lp
    su = None
    if "s3_uri" in row.index and pd.notna(row.get("s3_uri")):
        s = str(row["s3_uri"]).strip()
        su = s if s.startswith("s3://") else None
    tensors = load_chip_for_model(lp, target_hw, s3_uri=su)
    chip_id = str(row.get("chip_id", lp.stem))
    return lp, su, tensors, chip_id


def run_chip_forward(
    model: torch.nn.Module,
    device: torch.device,
    tensors: dict[str, torch.Tensor],
    mc_samples: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    xb = tensors["x"].unsqueeze(0)
    return predict_chip(model, xb, device, mc_samples=mc_samples)


def save_chip_outputs(
    out_dir: Path,
    chip_id: str,
    mean_p: torch.Tensor,
    var_p: torch.Tensor | None,
    mask: torch.Tensor,
    meta: dict[str, Any],
    *,
    source_path: Path | None,
    write_geotiff: bool,
    flat_output: bool = False,
) -> None:
    import json
    import math

    from src.modeling.aggregate import chip_aggregate

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, list):
            return [_json_safe(x) for x in obj]
        return obj
    from src.modeling.raster_export import write_prob_geotiffs

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in chip_id)[:200]
    sub = out_dir if flat_output else (out_dir / safe)
    sub.mkdir(parents=True, exist_ok=True)

    mp = mean_p.squeeze().cpu().numpy().astype(np.float32)
    vp: np.ndarray | None = None
    np.savez_compressed(sub / "mean_prob.npz", prob=mp)
    if var_p is not None:
        vp = var_p.squeeze().cpu().numpy().astype(np.float32)
        np.savez_compressed(sub / "var_prob.npz", var=vp)
    agg = chip_aggregate(mean_p, mask.unsqueeze(0), var_p)
    meta_out = _json_safe({**meta, "chip_id": chip_id, "aggregate": agg})
    (sub / "aggregate.json").write_text(json.dumps(meta_out, indent=2, allow_nan=False), encoding="utf-8")

    if write_geotiff and source_path is not None and source_path.exists():
        write_prob_geotiffs(source_path, mp, sub, "pred", var_prob_hw=vp)
