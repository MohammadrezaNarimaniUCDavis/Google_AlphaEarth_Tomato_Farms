"""Summarize LandIQ layers: geometry count, CRS, columns, and value counts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


def load_shapefile(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    return gpd.read_file(path)


def summarize_gdf(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    return {
        "n_rows": len(gdf),
        "crs": str(gdf.crs) if gdf.crs is not None else None,
        "columns": list(gdf.columns),
        "geom_type": gdf.geometry.geom_type.value_counts().to_dict(),
        "total_area_ha": float(gdf.geometry.area.sum() * 1e-4) if gdf.crs and gdf.crs.is_projected else None,
    }


def value_counts_for_columns(gdf: gpd.GeoDataFrame, columns: list[str], top_n: int = 30) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for col in columns:
        if col in gdf.columns:
            out[col] = gdf[col].value_counts(dropna=False).head(top_n)
    return out
