"""Map legend codes to columns in LandIQ attribute tables (e.g. tomato T15, T26)."""

from __future__ import annotations

import re
from typing import Sequence

import geopandas as gpd
import pandas as pd

from .years.landiq_2018 import TOMATO_CODES as TOMATO_CODES_2018

# Default Land IQ pattern: CROPTYP1, CROPTYP2, CROPTYP3 (up to three crop slots per polygon).
CROPTYP_COLUMN_PATTERN = re.compile(r"^CROPTYP\d+$", re.IGNORECASE)


def normalize_series_as_string(s: pd.Series) -> pd.Series:
    """Stringify values, strip whitespace; NaN becomes ``\"nan\"`` for comparison."""
    out = s.astype("string")
    out = out.str.strip()
    return out


def scan_columns_for_codes(
    gdf: gpd.GeoDataFrame,
    codes: Sequence[str],
    *,
    geometry_col: str = "geometry",
) -> pd.DataFrame:
    """Count rows where each column equals each code (after string strip).

    Returns a long table with columns ``column``, ``code``, ``count``.
    """
    codes_list = [str(c).strip() for c in codes]
    rows: list[dict[str, str | int]] = []
    for col in gdf.columns:
        if col == geometry_col:
            continue
        normalized = normalize_series_as_string(gdf[col])
        for code in codes_list:
            n = int((normalized == code).sum())
            if n > 0:
                rows.append({"column": col, "code": code, "count": n})
    return pd.DataFrame(rows)


def croptyp_column_names(
    gdf: gpd.GeoDataFrame,
    pattern: re.Pattern[str] | None = None,
) -> list[str]:
    """Return attribute columns that look like ``CROPTYP1``, ``CROPTYP2``, … (order preserved)."""
    rx = pattern or CROPTYP_COLUMN_PATTERN
    return [c for c in gdf.columns if c != "geometry" and rx.match(str(c))]


def scan_croptyp_columns_for_codes(
    gdf: gpd.GeoDataFrame,
    codes: Sequence[str],
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Count **T15** / **T26** (or other codes) in each ``CROPTYP*`` column only.

    Returns columns ``column``, ``code``, ``count`` (only rows with count > 0).
    """
    cols = columns or croptyp_column_names(gdf)
    codes_list = [str(c).strip() for c in codes]
    rows: list[dict[str, str | int]] = []
    for col in cols:
        normalized = normalize_series_as_string(gdf[col])
        for code in codes_list:
            n = int((normalized == code).sum())
            if n > 0:
                rows.append({"column": col, "code": code, "count": n})
    return pd.DataFrame(rows)


def tomato_mask_any_croptyp(
    gdf: gpd.GeoDataFrame,
    codes: Sequence[str],
    *,
    columns: list[str] | None = None,
) -> pd.Series:
    """Boolean mask: True if **any** listed ``CROPTYP*`` column equals one of ``codes``."""
    cols = columns or croptyp_column_names(gdf)
    if not cols:
        return pd.Series(False, index=gdf.index)
    codes_set = {str(c).strip() for c in codes}
    stacked = pd.concat(
        [normalize_series_as_string(gdf[c]).isin(codes_set) for c in cols],
        axis=1,
    )
    return stacked.any(axis=1)


def summarize_tomato_croptyp_coverage(
    gdf: gpd.GeoDataFrame,
    codes: Sequence[str],
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """One row per ``CROPTYP*`` column with counts for each code (including zeros for missing codes)."""
    cols = columns or croptyp_column_names(gdf)
    codes_list = [str(c).strip() for c in codes]
    out_rows: list[dict[str, str | int]] = []
    for col in cols:
        normalized = normalize_series_as_string(gdf[col])
        row: dict[str, str | int] = {"column": col}
        for code in codes_list:
            row[code] = int((normalized == code).sum())
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def attribute_table_overview(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """One row per non-geometry column: dtype, non-null count, n_unique."""
    rows = []
    for col in gdf.columns:
        if col == "geometry":
            continue
        s = gdf[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def dwr_group_from_code(code: str) -> str:
    """Map a LandIQ/DWR crop code into a coarse group label.

    Examples:
    - ``T15`` -> ``T``
    - ``G6`` -> ``G``
    - ``YP`` -> ``YP``
    - ``****`` / empty -> ``UNK``
    """
    if code is None:
        return "UNK"
    s = str(code).strip()
    if not s or s == "****":
        return "UNK"
    if s.upper() == "YP":
        return "YP"
    # Most codes are like "<LETTER><digits>" (e.g. C1, F10, V2). Group by the leading letter.
    return s[0].upper()
