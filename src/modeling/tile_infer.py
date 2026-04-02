"""Sliding-window inference on a large multi-band GeoTIFF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

from src.modeling.dataset import _resize_stack
from src.modeling.infer_io import predict_chip


def _read_padded_patch(
    src: rasterio.io.DatasetReader,
    y: int,
    x: int,
    ph: int,
    pw: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read ``C, ph, pw`` from ``src``, zero-pad if near edges."""
    H, W = src.height, src.width
    arr = np.zeros((src.count, ph, pw), dtype=np.float32)
    valid = np.zeros((ph, pw), dtype=np.float32)
    y1, x1 = min(y + ph, H), min(x + pw, W)
    h0, w0 = max(0, y), max(0, x)
    rh, rw = y1 - h0, x1 - x0
    if rh <= 0 or rw <= 0:
        return arr, valid
    win = Window(x0, y0, rw, rh)
    sub = src.read(out_dtype="float32", window=win)
    oy, ox = h0 - y, w0 - x
    arr[:, oy : oy + rh, ox : ox + rw] = sub
    vsub = np.isfinite(sub).all(axis=0).astype(np.float32)
    valid[oy : oy + rh, ox : ox + rw] = vsub
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, valid


@torch.no_grad()
def infer_large_geotiff(
    model: torch.nn.Module,
    device: torch.device,
    raster_path: str | Path,
    target_hw: tuple[int, int],
    *,
    tile_h: int | None = None,
    tile_w: int | None = None,
    overlap: int = 32,
    mc_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Sliding windows of size ``tile_h × tile_w`` (default = ``target_hw``), overlap, average blend.

    Returns ``mean_prob`` (H,W) in **original raster grid**, optional ``var_prob`` same shape,
    and meta dict with crs WKT and transform tuple.
    """
    th, tw = target_hw
    ph = tile_h or th
    pw = tile_w or tw
    step_y = max(1, ph - overlap)
    step_x = max(1, pw - overlap)

    with rasterio.open(raster_path) as src:
        H, W = src.height, src.width
        crs = src.crs
        transform = src.transform
        b = src.bounds

        acc = np.zeros((H, W), dtype=np.float64)
        wsum = np.zeros((H, W), dtype=np.float64)
        var_acc: np.ndarray | None = None
        var_wsum: np.ndarray | None = None
        if mc_samples > 0:
            var_acc = np.zeros((H, W), dtype=np.float64)
            var_wsum = np.zeros((H, W), dtype=np.float64)

        for y in range(0, H, step_y):
            for x in range(0, W, step_x):
                arr, valid_hw = _read_padded_patch(src, y, x, ph, pw)
                arr_r, valid_r = _resize_stack(arr, valid_hw, th, tw)
                x_t = torch.from_numpy(arr_r).float().unsqueeze(0)
                m_t = torch.from_numpy(valid_r).float().unsqueeze(0).unsqueeze(0)
                mean_p, var_p = predict_chip(model, x_t, device, mc_samples=mc_samples)
                # Resize prediction back to patch grid for accumulation
                mp = mean_p.float().squeeze(0)  # 1, th, tw
                mp_up = F.interpolate(mp.unsqueeze(0), size=(ph, pw), mode="bilinear", align_corners=False)
                mp_up = mp_up.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
                wt = valid_hw.astype(np.float64)
                if wt.sum() < 1:
                    continue
                y1, x1 = min(y + ph, H), min(x + pw, W)
                h0, w0 = max(0, y), max(0, x)
                sl_y = slice(h0, y1)
                sl_x = slice(w0, x1)
                oy, ox = h0 - y, w0 - x
                rh, rw = y1 - h0, x1 - w0
                patch_p = mp_up[oy : oy + rh, ox : ox + rw]
                patch_w = wt[oy : oy + rh, ox : ox + rw]
                acc[sl_y, sl_x] += patch_p * patch_w
                wsum[sl_y, sl_x] += patch_w

                if var_p is not None and var_acc is not None and var_wsum is not None:
                    vp = var_p.float().squeeze(0)
                    vp_up = F.interpolate(vp.unsqueeze(0), size=(ph, pw), mode="bilinear", align_corners=False)
                    vp_up = vp_up.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
                    patch_v = vp_up[oy : oy + rh, ox : ox + rw]
                    var_acc[sl_y, sl_x] += patch_v * patch_w
                    var_wsum[sl_y, sl_x] += patch_w

        out = np.divide(acc, np.maximum(wsum, 1e-8), dtype=np.float32)
        out_var: np.ndarray | None = None
        if var_acc is not None and var_wsum is not None:
            out_var = np.divide(var_acc, np.maximum(var_wsum, 1e-8)).astype(np.float32)

    meta = {
        "crs": crs.to_wkt() if crs else None,
        "transform_gdal": transform.to_gdal(),
        "bounds": (b.left, b.bottom, b.right, b.top),
        "height": H,
        "width": W,
    }
    return out, out_var, meta


def write_raster_from_meta(
    data_hw: np.ndarray,
    meta: dict[str, Any],
    out_path: Path,
) -> None:
    """Write single-band float32 GeoTIFF using transform from ``meta``."""
    from rasterio.transform import Affine

    H, W = data_hw.shape
    data_hw = np.asarray(data_hw, dtype=np.float32)
    # meta['transform'] from to_gdal() - use Affine.from_gdal
    transform = Affine.from_gdal(*meta["transform_gdal"])
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    crs_wkt = meta.get("crs")
    if crs_wkt:
        from rasterio.crs import CRS

        profile["crs"] = CRS.from_wkt(crs_wkt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data_hw[np.newaxis, :, :])
