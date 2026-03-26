"""Google Earth Engine helpers for AlphaEarth / Satellite Embedding annual layers."""

from __future__ import annotations

import json
import os
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import numbers

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from shapely import force_2d
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

# Official collection: 64 bands A00–A63, ~10 m, calendar-year composites.
# See https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
DEFAULT_EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
# GEE catalog native resolution for GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (~10 m).
NATIVE_ALPHA_EARTH_ANNUAL_SCALE_M = 10

# Catalog annual embeddings start at 2017.
EMBEDDING_YEAR_MIN = 2017
# Last year with full global coverage may lag catalog updates; raise if EE returns no tiles.
EMBEDDING_YEAR_MAX = 2024


def validate_embedding_year(year: int, lo: int = EMBEDDING_YEAR_MIN, hi: int = EMBEDDING_YEAR_MAX) -> None:
    if year < lo or year > hi:
        raise ValueError(
            f"embedding_year {year} is outside [{lo}, {hi}] for {DEFAULT_EMBEDDING_COLLECTION}. "
            "Use 2017+; align with your LandIQ survey year when both exist (e.g. 2018 + 2018)."
        )


def shapely_to_ee_geometry(geom) -> ee.Geometry:
    """GeoJSON-style geometry for Earth Engine (expects WGS84 coordinates).

    Strips Z/M with ``force_2d``: EE's client rejects GeoJSON where each vertex
    has an odd-length coordinate tuple (e.g. lon/lat/0 from some GPKG sources).
    """
    if geom is None or geom.is_empty:
        raise ValueError("Cannot convert empty or missing geometry to Earth Engine.")
    return ee.Geometry(mapping(force_2d(geom)))


def annual_embedding_mosaic(
    year: int,
    collection_id: str = DEFAULT_EMBEDDING_COLLECTION,
) -> ee.Image:
    """Full annual mosaic for ``year`` (no spatial filter, no server round-trip).

    For global annual composites such as ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL``,
    pixels over any location match ``filterBounds(geom).mosaic()`` for that ``geom``.
    Build **once**, then ``.clip(…)`` per polygon to avoid a ``getInfo()`` per feature.
    """
    validate_embedding_year(year)
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    return (
        ee.ImageCollection(collection_id)
        .filterDate(start, end)
        .mosaic()
        .select([f"A{i:02d}" for i in range(64)])
    )


def annual_embedding_over_geometry(
    year: int,
    geometry: ee.Geometry,
    collection_id: str = DEFAULT_EMBEDDING_COLLECTION,
    *,
    check_tiles: bool = True,
) -> ee.Image:
    """Mosaic annual embedding tiles intersecting ``geometry`` for the given calendar year.

    If ``check_tiles`` is False, returns the same mosaic as :func:`annual_embedding_mosaic`
    clipped to ``geometry`` (no ``getInfo`` call — use for bulk exports after a one-time check).
    """
    if not check_tiles:
        return annual_embedding_mosaic(year, collection_id).clip(geometry)
    validate_embedding_year(year)
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    ic = ee.ImageCollection(collection_id).filterDate(start, end).filterBounds(geometry)
    count = ic.size().getInfo()
    if count == 0:
        raise RuntimeError(
            f"No embedding images for {year} intersecting the geometry "
            f"(collection={collection_id}). Check bounds and year."
        )
    return ic.mosaic().select([f"A{i:02d}" for i in range(64)])


def embedding_band_names() -> list[str]:
    return [f"A{i:02d}" for i in range(64)]


def warn_unless_native_embedding_scale(scale_m: int, collection_id: str) -> None:
    """If using the default AlphaEarth annual collection, ``scale_m`` must be 10 for native resolution."""
    if collection_id == DEFAULT_EMBEDDING_COLLECTION and scale_m != NATIVE_ALPHA_EARTH_ANNUAL_SCALE_M:
        warnings.warn(
            f"scale_m={scale_m} resamples the export away from catalog native resolution "
            f"({NATIVE_ALPHA_EARTH_ANNUAL_SCALE_M} m) for {DEFAULT_EMBEDDING_COLLECTION}. "
            "Use scale_m=10 for full native pixel grid (larger files are expected).",
            UserWarning,
            stacklevel=2,
        )


def apply_geotiff_polygon_mask_from_geojson(
    path_str: str,
    geojson: dict[str, Any],
    *,
    all_touched: bool = True,
) -> None:
    """Apply :func:`_mask_geotiff_outside_polygon` from WGS84 GeoJSON (thread-pool friendly)."""
    _mask_geotiff_outside_polygon(Path(path_str), shape(geojson), all_touched=all_touched)


def _geojson_to_python_floats(obj: Any) -> Any:
    """Recursively convert numpy scalars in GeoJSON-like structures for rasterio/shapely."""
    if isinstance(obj, dict):
        return {k: _geojson_to_python_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_geojson_to_python_floats(v) for v in obj]
    if isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        return float(obj)
    return obj


def apply_geotiff_polygon_mask_task(payload: tuple[str, dict[str, Any], bool]) -> None:
    """Single-arg wrapper for :class:`concurrent.futures.ThreadPoolExecutor` / sequential masking."""
    path_str, geojson, all_touched = payload
    apply_geotiff_polygon_mask_from_geojson(path_str, geojson, all_touched=all_touched)


def _mask_geotiff_outside_polygon(
    out_path: Path,
    polygon_wgs84: BaseGeometry,
    *,
    all_touched: bool = True,
) -> None:
    """Set pixels outside the polygon to NaN (float32), keeping a square GeoTIFF extent.

    Geometry is assumed WGS84 (EPSG:4326); it is reprojected to the raster CRS if needed.
    ``all_touched`` matches GDAL/rasterio semantics: any pixel touching the polygon edge
    or interior is kept (appropriate for ~10 m pixels along field boundaries).
    """
    g = force_2d(polygon_wgs84)
    if g.is_empty:
        raise ValueError("Polygon geometry is empty; cannot build mask.")
    geojson_wgs84 = mapping(g)

    with rasterio.open(out_path) as src:
        if src.crs is None:
            raise ValueError("GeoTIFF has no CRS; cannot align polygon mask.")
        dst_crs = src.crs
        wgs84 = CRS.from_epsg(4326)
        # Use == (rasterio CRS); .equals() is not on all rasterio versions.
        if dst_crs != wgs84:
            geom_for_raster = transform_geom(wgs84, dst_crs, geojson_wgs84)
        else:
            geom_for_raster = geojson_wgs84

        geom_for_raster = _geojson_to_python_floats(geom_for_raster)
        geom_shape = shape(geom_for_raster)

        out_shape_hw = (src.height, src.width)
        # True = pixel does NOT intersect polygon → we will NaN those cells.
        outside = geometry_mask(
            [geom_shape],
            out_shape=out_shape_hw,
            transform=src.transform,
            all_touched=all_touched,
            invert=False,
        )
        inside = ~outside

        data = src.read()
        profile = src.profile.copy()

        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
            profile.update(dtype=rasterio.float32)

        out = np.empty_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            out[i] = np.where(inside, data[i], np.nan)

        profile.update(dtype=rasterio.float32, nodata=None)

    tmp_path = out_path.with_suffix(out_path.suffix + ".masktmp")
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(out)
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def download_clipped_geotiff(
    image: ee.Image,
    region: ee.Geometry,
    out_path: Path,
    scale_m: int = 10,
    crs: str = "EPSG:4326",
    max_pixels: float = 1e9,
    *,
    polygon_wgs84: BaseGeometry | None = None,
    rasterize_all_touched: bool = True,
    skip_local_polygon_mask: bool = False,
) -> None:
    """Download a multi-band GeoTIFF via Earth Engine (pilot-scale regions only).

    Earth Engine ``scale`` is ``scale_m`` meters — use ``10`` for native AlphaEarth annual
    resolution (no resampling vs catalog).

    If ``polygon_wgs84`` is set, the image is clipped to the polygon's **bounding box**
    for export (square pixel grid aligned to ``scale_m`` / ``crs``), then pixels outside
    the polygon are set to **NaN** locally using ``rasterio.features.geometry_mask`` /
    ``all_touched=rasterize_all_touched`` (include partial edge pixels at native resolution).

    Set ``skip_local_polygon_mask=True`` to download only (full bbox values); mask later
    e.g. with :func:`apply_geotiff_polygon_mask_task` in a process pool.

    If ``polygon_wgs84`` is omitted, behavior matches the older path: ``clip`` to
    ``region`` only on the EE side (masking may not match polygon footprint in the file).
    """
    if polygon_wgs84 is not None:
        export_region = region.bounds()
        clipped = image.clip(export_region)
    else:
        export_region = region
        clipped = image.clip(region)

    params: dict[str, Any] = {
        "scale": scale_m,
        "crs": crs,
        "region": export_region,
        "format": "GEO_TIFF",
        "maxPixels": max_pixels,
    }
    url = clipped.getDownloadURL(params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)

    if polygon_wgs84 is not None and not skip_local_polygon_mask:
        _mask_geotiff_outside_polygon(
            out_path,
            polygon_wgs84,
            all_touched=rasterize_all_touched,
        )


def write_pilot_manifest(
    path: Path,
    *,
    source_gpkg: str,
    landiq_survey_year: int | None,
    embedding_year: int,
    collection_id: str,
    scale_m: int,
    crs_export: str,
    polygon_indices: list[int | str],
    output_files: list[str],
    band_names: list[str],
    polygon_mask_all_touched: bool | None = None,
) -> None:
    export: dict[str, Any] = {
        "format": "GeoTIFF",
        "scale_m": scale_m,
        "crs": crs_export,
        "bands": band_names,
    }
    if polygon_mask_all_touched is not None:
        export["polygon_mask"] = {
            "outside_polygon_values": "NaN (float32)",
            "rasterio_geometry_mask_all_touched": polygon_mask_all_touched,
            "note": "Square extent from EE bbox; pixels outside the LandIQ polygon set to NaN (geometry_mask, all_touched).",
        }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "attribution": (
            'The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind. '
            "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL"
        ),
        "source_tomato_gpkg": source_gpkg,
        "landiq_survey_year": landiq_survey_year,
        "embedding_calendar_year": embedding_year,
        "note": (
            "LandIQ survey year and embedding calendar year should match when both exist in the catalog "
            "(embeddings start 2017)."
        ),
        "gee_collection": collection_id,
        "export": export,
        "polygon_source_indices": polygon_indices,
        "outputs": output_files,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
