"""Filter LandIQ polygons to tomato crops. Configure column and values in paths config.

Row filter only: every **attribute column** (County, Acres, CROPTYP*, etc.) and **geometry**
are **kept unchanged** for polygons that match. Non-tomato polygons are **dropped** as whole
rows — we do **not** strip fields from tomato rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ..utils.paths import (
    REPO_ROOT,
    landiq_tomato_gpkg_path,
    load_paths_config,
    resolve_landiq_shapefile_path,
)


def _mask_tomato_values(series: pd.Series, tomato_values: list[str | int | float]) -> pd.Series:
    normalized = series.astype("string").str.strip()
    targets = {str(v).strip() for v in tomato_values}
    return normalized.isin(targets)


def filter_tomato(
    gdf: gpd.GeoDataFrame,
    crop_column: str,
    tomato_values: list[str | int | float],
) -> gpd.GeoDataFrame:
    if crop_column not in gdf.columns:
        raise KeyError(f"Column {crop_column!r} not in GeoDataFrame columns: {list(gdf.columns)}")
    mask = _mask_tomato_values(gdf[crop_column], tomato_values)
    return gdf.loc[mask].copy()


def filter_tomato_any_column(
    gdf: gpd.GeoDataFrame,
    crop_columns: list[str],
    tomato_values: list[str | int | float],
) -> gpd.GeoDataFrame:
    """Keep rows where **any** of ``crop_columns`` matches one of ``tomato_values`` (string-stripped)."""
    if not crop_columns:
        raise ValueError("crop_columns must be non-empty")
    masks: list[pd.Series] = []
    for col in crop_columns:
        if col not in gdf.columns:
            raise KeyError(f"Column {col!r} not in GeoDataFrame columns: {list(gdf.columns)}")
        masks.append(_mask_tomato_values(gdf[col], tomato_values))
    combined = pd.concat(masks, axis=1).any(axis=1)
    return gdf.loc[combined].copy()


def filter_tomatoes_from_landiq_config(
    gdf: gpd.GeoDataFrame,
    landiq_cfg: dict,
) -> gpd.GeoDataFrame:
    """Use ``landiq.crop_columns`` (OR) if set, else ``landiq.crop_column`` (single).

    Returns a **subset of rows** with the **same columns** as ``gdf`` (full attributes + geometry).
    """
    tomato_vals = landiq_cfg.get("tomato_values") or []
    if not tomato_vals:
        raise ValueError("landiq.tomato_values must be non-empty")
    cols = landiq_cfg.get("crop_columns")
    if cols and isinstance(cols, list) and len(cols) > 0:
        return filter_tomato_any_column(gdf, cols, tomato_vals)
    col = landiq_cfg.get("crop_column")
    if not col:
        raise ValueError("Set landiq.crop_columns (list) or landiq.crop_column (string) in config")
    return filter_tomato(gdf, col, tomato_vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter LandIQ shapes to tomato polygons.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Input vector path (.shp, .gpkg, .geojson). Overrides config if set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (.gpkg recommended). Overrides config if set.",
    )
    args = parser.parse_args()

    cfg = load_paths_config()
    proj = cfg.get("project_root", ".")
    root = Path(proj).resolve() if proj != "." else REPO_ROOT

    landiq_cfg = cfg.get("landiq", {})
    tomato_vals = landiq_cfg.get("tomato_values") or []
    crop_cols = landiq_cfg.get("crop_columns")
    crop_col = landiq_cfg.get("crop_column")
    if not tomato_vals or (not crop_col and not (crop_cols and isinstance(crop_cols, list) and len(crop_cols) > 0)):
        raise SystemExit(
            "Set landiq.tomato_values and either landiq.crop_columns (list, OR) or "
            "landiq.crop_column (single) in configs/paths.local.yaml."
        )

    if args.input:
        in_path = args.input.resolve()
    else:
        try:
            in_path = resolve_landiq_shapefile_path(cfg, root)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(str(e)) from e

    if args.output:
        out_path = args.output.resolve()
    else:
        out_path = landiq_tomato_gpkg_path(cfg, root)

    gdf = gpd.read_file(in_path)
    tomato = filter_tomatoes_from_landiq_config(gdf, landiq_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tomato.to_file(out_path, driver="GPKG")
    print(f"Wrote {len(tomato)} tomato polygons (all attribute columns preserved) -> {out_path}")


if __name__ == "__main__":
    main()
