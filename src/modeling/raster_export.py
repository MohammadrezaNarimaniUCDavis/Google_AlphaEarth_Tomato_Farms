"""Write model outputs as GeoTIFFs aligned to source raster bounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_bounds


def write_prob_geotiffs(
    source_raster: str | Path,
    mean_prob_hw: np.ndarray,
    out_dir: Path,
    stem: str,
    *,
    var_prob_hw: np.ndarray | None = None,
) -> list[Path]:
    """Map ``mean_prob_hw`` (H,W) to **same geographic bounds** as ``source_raster`` (model resize is geometric)."""
    mean_prob_hw = np.asarray(mean_prob_hw, dtype=np.float32)
    h, w = mean_prob_hw.shape
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with rasterio.open(source_raster) as src:
        b = src.bounds
        crs = src.crs
        transform = from_bounds(b.left, b.bottom, b.right, b.top, w, h)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    p_mean = out_dir / f"{stem}_mean_prob.tif"
    with rasterio.open(p_mean, "w", **profile) as dst:
        dst.write(mean_prob_hw[np.newaxis, :, :])
    written.append(p_mean)
    if var_prob_hw is not None:
        vp = np.asarray(var_prob_hw, dtype=np.float32)
        p_var = out_dir / f"{stem}_var_prob.tif"
        with rasterio.open(p_var, "w", **profile) as dst:
            dst.write(vp[np.newaxis, :, :])
        written.append(p_var)
    return written
