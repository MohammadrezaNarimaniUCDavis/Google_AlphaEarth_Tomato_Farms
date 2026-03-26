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


def resolve_raster_path_for_year(raster_root: Path, year: int) -> Path | None:
    """Find a GeoTIFF for ``year``.

    Search order:

    1. ``raster_root / str(year) /`` — any ``*.tif`` (shallow glob, then recursive).
    2. ``raster_root / *{year}*.tif`` (top level only).
    """
    if not raster_root.is_dir():
        return None
    sub = raster_root / str(year)
    if sub.is_dir():
        shallow = sorted(sub.glob("*.tif"))
        if shallow:
            return shallow[0]
        deep = sorted(sub.rglob("*.tif"))
        if deep:
            return deep[0]
    top = sorted(raster_root.glob(f"*{year}*.tif"))
    return top[0] if top else None


def load_raster_for_year(year: int, raster_root: Path) -> "rasterio.io.DatasetReader | None":
    """Open the first matching GeoTIFF for ``year`` under ``raster_root`` (see ``resolve_raster_path_for_year``)."""
    if rasterio is None:
        return None
    p = resolve_raster_path_for_year(raster_root, year)
    if p is None:
        return None
    return rasterio.open(p)


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
