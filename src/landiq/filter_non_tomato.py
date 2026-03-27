"""Filter LandIQ polygons to non-tomato crops (binary negatives).

Rules:
- Exclude any polygon where ANY of the configured crop slot columns contains
  a tomato code (by default T15/T26).
- From the remaining non-tomato polygons, sample exactly the same count as
  the tomato set (balanced over a simple DWR group derived from the first
  character of each code).

This produces a GeoPackage of negative (non-tomato) polygons that keeps the
same schema as the input shapefile.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .legend_codes import dwr_group_from_code, tomato_mask_any_croptyp
from ..utils.paths import (
    REPO_ROOT,
    landiq_non_tomato_gpkg_path,
    load_paths_config,
    resolve_landiq_shapefile_path,
)


def _dwr_group_for_row(row: pd.Series, crop_columns: list[str], tomato_values: list[str | int | float]) -> str:
    """Assign a group for a non-tomato row by scanning CROPTYP1..CROPTYPn."""
    tomato_set = {str(v).strip() for v in tomato_values}
    for col in crop_columns:
        if col not in row:
            continue
        v = row[col]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v).strip()
        if not s:
            continue
        if s in tomato_set:
            # Should not happen for non-tomato rows, but keep safe.
            continue
        return dwr_group_from_code(s)
    return "UNK"


def _balanced_sample_by_group(
    non_df: gpd.GeoDataFrame,
    groups: pd.Series,
    *,
    target_n: int,
    seed: int,
) -> gpd.GeoDataFrame:
    if target_n > len(non_df):
        raise ValueError(f"target_n={target_n} > available non-tomato rows={len(non_df)}")

    # Group sizes (descending).
    vc = groups.value_counts()
    group_labels = vc.index.tolist()
    k = len(group_labels)
    if k == 0:
        raise ValueError("No non-tomato rows to sample.")

    base = target_n // k
    rem = target_n - base * k

    # Deterministic group order: by size desc, then label asc.
    ordered = sorted(group_labels, key=lambda g: (-int(vc[g]), str(g)))

    selected_idx: list[int] = []
    for i, g in enumerate(ordered):
        take = base + (1 if i < rem else 0)
        if take <= 0:
            continue
        idx = non_df.index[groups == g]
        if len(idx) == 0:
            continue
        take = min(take, len(idx))
        # Use per-group deterministic seed.
        sample_idx = (
            non_df.loc[idx].sample(n=take, random_state=seed + i).index.tolist()
            if take < len(idx)
            else idx.tolist()
        )
        selected_idx.extend(sample_idx)

    # Fill any shortfall (due to groups with fewer samples than their target).
    if len(selected_idx) < target_n:
        remaining = non_df.loc[~non_df.index.isin(selected_idx)]
        remaining_groups = groups.loc[remaining.index].value_counts().index.tolist()
        for g in remaining_groups:
            need = target_n - len(selected_idx)
            if need <= 0:
                break
            idx = remaining.index[groups.loc[remaining.index] == g]
            if len(idx) == 0:
                continue
            take = min(need, len(idx))
            sample_idx = non_df.loc[idx].sample(n=take, random_state=seed + 999).index.tolist()
            selected_idx.extend(sample_idx)
            remaining = non_df.loc[~non_df.index.isin(selected_idx)]

    selected_idx = selected_idx[:target_n]
    out = non_df.loc[selected_idx].copy()
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def _area_series_m2(
    gdf: gpd.GeoDataFrame,
    *,
    preferred_columns: list[str] | None = None,
) -> pd.Series:
    """Return per-row area in m^2.

    Prefers an existing attribute column like LandIQ's ``Shape_STAr`` when present.
    Falls back to geometry area computed in EPSG:3310 (California Albers) when needed.
    """
    cols = preferred_columns or ["Shape_STAr", "Shape_Area", "SHAPE_Area", "AREA"]
    for c in cols:
        if c in gdf.columns:
            s = pd.to_numeric(gdf[c], errors="coerce")
            if s.notna().any():
                return s

    # Fallback: compute from geometry (projected CRS required).
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS and no usable area attribute column.")
    return gdf.to_crs(3310).geometry.area


def _drop_huge_polygons(
    non_df: gpd.GeoDataFrame,
    *,
    max_area_m2: float | None,
    max_area_quantile: float | None,
    area_preferred_columns: list[str] | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Drop extreme area outliers before sampling negatives."""
    area_m2 = _area_series_m2(non_df, preferred_columns=area_preferred_columns)

    info: dict[str, Any] = {
        "area_column_used": next((c for c in (area_preferred_columns or ["Shape_STAr"]) if c in non_df.columns), None),
        "area_non_null": int(area_m2.notna().sum()),
        "dropped_by_abs": 0,
        "dropped_by_q": 0,
        "threshold_abs_m2": max_area_m2,
        "threshold_q": max_area_quantile,
        "threshold_q_m2": None,
    }

    keep = area_m2.notna()

    if max_area_quantile is not None:
        if not (0 < float(max_area_quantile) < 1):
            raise ValueError("max_area_quantile must be between 0 and 1 (e.g. 0.995)")
        thr_q = float(area_m2.quantile(float(max_area_quantile)))
        info["threshold_q_m2"] = float(thr_q)
        before = int(keep.sum())
        keep &= area_m2 <= thr_q
        info["dropped_by_q"] = before - int(keep.sum())

    if max_area_m2 is not None:
        before = int(keep.sum())
        keep &= area_m2 <= float(max_area_m2)
        info["dropped_by_abs"] = before - int(keep.sum())

    return non_df.loc[keep].copy(), info


def filter_non_tomatoes_from_landiq_config(
    gdf: gpd.GeoDataFrame,
    landiq_cfg: dict[str, Any],
    *,
    target_n: int | None = None,
    seed: int = 42,
    max_area_m2: float | None = 10_000_000.0,
    max_area_quantile: float | None = 0.995,
) -> gpd.GeoDataFrame:
    """Return a balanced non-tomato (negative) GeoDataFrame.

    The return value is a subset of ``gdf`` rows (all attributes + geometry preserved).
    """
    tomato_vals = landiq_cfg.get("tomato_values") or []
    if not tomato_vals:
        raise ValueError("landiq.tomato_values must be non-empty (e.g. [\"T15\",\"T26\"])")

    crop_cols: list[str] | None = landiq_cfg.get("crop_columns")
    if crop_cols and isinstance(crop_cols, list):
        crop_columns = [str(c) for c in crop_cols if c]
    else:
        c = landiq_cfg.get("crop_column")
        crop_columns = [str(c)] if c else []

    if not crop_columns:
        raise ValueError("Set landiq.crop_columns (list) or landiq.crop_column in config.")
    for c in crop_columns:
        if c not in gdf.columns:
            raise KeyError(f"Crop column {c!r} not found in GeoDataFrame.")

    tomato_mask = tomato_mask_any_croptyp(gdf, tomato_vals, columns=crop_columns)
    tomato_n = int(tomato_mask.sum())

    non_df = gdf.loc[~tomato_mask].copy()
    if len(non_df) == 0:
        raise ValueError("No non-tomato rows available after excluding tomato codes.")

    if target_n is None:
        target_n = tomato_n

    non_df, drop_info = _drop_huge_polygons(
        non_df,
        max_area_m2=max_area_m2,
        max_area_quantile=max_area_quantile,
        area_preferred_columns=["Shape_STAr", "Shape_Area", "SHAPE_Area"],
    )
    if target_n > len(non_df):
        raise ValueError(
            f"After dropping huge polygons, available non-tomato rows={len(non_df)} < target_n={target_n}. "
            f"Relax max_area_m2/max_area_quantile. Details: {drop_info}"
        )

    # Group assignment for negatives.
    groups = non_df.apply(lambda r: _dwr_group_for_row(r, crop_columns, tomato_vals), axis=1)
    return _balanced_sample_by_group(non_df, groups, target_n=target_n, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter LandIQ shapes to non-tomato polygons (binary negatives).")
    parser.add_argument("--input", type=Path, help="Input vector path (.shp, .gpkg, .geojson). Overrides config.")
    parser.add_argument("--output", type=Path, help="Output path (.gpkg recommended). Overrides config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling.")
    parser.add_argument(
        "--target_n",
        type=int,
        default=None,
        help="Number of negatives to sample (default: equal to tomato count).",
    )
    args = parser.parse_args()

    cfg = load_paths_config()
    landiq_cfg = cfg.get("landiq", {})

    root = Path(cfg.get("project_root", "."))
    root = root.resolve() if str(root) != "." else REPO_ROOT

    if args.input:
        in_path = args.input.resolve()
    else:
        in_path = resolve_landiq_shapefile_path(cfg, root)

    if args.output:
        out_path = args.output.resolve()
    else:
        out_path = landiq_non_tomato_gpkg_path(cfg, root)

    gdf = gpd.read_file(in_path)
    out_gdf = filter_non_tomatoes_from_landiq_config(
        gdf,
        landiq_cfg,
        target_n=args.target_n,
        seed=args.seed,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(out_path, driver="GPKG")

    tomato_vals = landiq_cfg.get("tomato_values") or []
    crop_cols = landiq_cfg.get("crop_columns")
    if crop_cols and isinstance(crop_cols, list):
        crop_columns = [str(c) for c in crop_cols if c]
    else:
        c = landiq_cfg.get("crop_column")
        crop_columns = [str(c)] if c else []
    tomato_mask = tomato_mask_any_croptyp(gdf, tomato_vals, columns=crop_columns)

    print(f"Wrote {len(out_gdf)} non-tomato polygons -> {out_path}")
    print(f"Input rows: {len(gdf)} | tomato rows excluded: {int(tomato_mask.sum())}")


if __name__ == "__main__":
    main()

