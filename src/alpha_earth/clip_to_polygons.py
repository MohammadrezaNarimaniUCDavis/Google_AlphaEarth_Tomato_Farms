"""Clip Alpha Earth layers to polygons for a range of years.

Replace `load_raster_for_year` with your actual data source (Earth Engine export,
local COGs, API, etc.). This module only defines the intended interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import geopandas as gpd

# Optional: rasterio for local rasters
try:
    import rasterio
    from rasterio.mask import mask as rio_mask
except ImportError:  # pragma: no cover
    rasterio = None
    rio_mask = None


def load_raster_for_year(year: int, raster_root: Path) -> "rasterio.io.DatasetReader | None":
    """Return an open rasterio dataset for `year`, or None if not implemented.

    Expected layout example: raster_root / f"alpha_earth_{year}.tif"
    """
    if rasterio is None:
        return None
    candidates = sorted(raster_root.glob(f"*{year}*.tif"))
    if not candidates:
        return None
    return rasterio.open(candidates[0])


def clip_raster_to_gdf(
    raster_ds: "rasterio.io.DatasetReader",
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    all_touched: bool = False,
) -> None:
    """Write a GeoTIFF clipped to the union of geometries in WGS84 or raster CRS."""
    if rio_mask is None:
        raise RuntimeError("rasterio is required for clip_raster_to_gdf")
    gdf = gdf.to_crs(raster_ds.crs)
    shapes = (geom.__geo_interface__ for geom in gdf.geometry)
    data, transform = rio_mask(raster_ds, shapes, all_touched=all_touched, crop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = raster_ds.profile.copy()
    profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
    with rasterio.open(out_path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            for i in range(1, data.shape[0] + 1):
                dst.write(data[i - 1], i)


def clip_years(
    polygons: gpd.GeoDataFrame,
    years: list[int],
    raster_root: Path,
    out_dir: Path,
    naming: Callable[[int, int], str] | None = None,
) -> list[Path]:
    """For each year, clip the year's raster to all polygons; returns output paths.

    `naming(year, index)` defaults to ``f\"clip_{year}_{index}.tif\"``.
    """
    written: list[Path] = []
    name_fn = naming or (lambda y, i: f"clip_{y}_{i}.tif")
    for year in years:
        ds = load_raster_for_year(year, raster_root)
        if ds is None:
            continue
        try:
            out_path = out_dir / name_fn(year, 0)
            clip_raster_to_gdf(ds, polygons, out_path)
            written.append(out_path)
        finally:
            ds.close()
    return written
