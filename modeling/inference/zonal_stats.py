#!/usr/bin/env python3
"""Per-polygon zonal stats on a probability (or variance) GeoTIFF — multi-farm summaries.

Requires **fiona** (vector read). Install: ``pip install fiona`` (often alongside rasterio).

Example::

    python modeling/inference/zonal_stats.py \\
      --raster outputs/predictions/region_prob.tif \\
      --vector data/derived/some_farms.gpkg \\
      --id-field farm_id \\
      --out-csv outputs/predictions/zonal_by_farm.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import rasterio
from rasterio.mask import mask

try:
    import fiona
except ImportError as e:
    raise SystemExit("Install fiona: pip install fiona") from e


def main() -> None:
    ap = argparse.ArgumentParser(description="Zonal stats per polygon on raster")
    ap.add_argument("--raster", type=Path, required=True)
    ap.add_argument("--vector", type=Path, required=True, help="GeoPackage, GeoJSON, or shapefile path")
    ap.add_argument("--id-field", type=str, default="id", help="Feature property for row id (fallback: feature index)")
    ap.add_argument("--band", type=int, default=1, help="Raster band index (1-based)")
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    rpath = args.raster if args.raster.is_absolute() else _REPO / args.raster
    vpath = args.vector if args.vector.is_absolute() else _REPO / args.vector
    out_csv = args.out_csv if args.out_csv.is_absolute() else _REPO / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float]] = []
    with rasterio.open(rpath) as src:
        with fiona.open(vpath) as vec:
            for i, feat in enumerate(vec):
                geom = feat["geometry"]
                if geom is None:
                    continue
                props = feat.get("properties") or {}
                fid = props.get(args.id_field, i)
                try:
                    arr, _tr = mask(
                        src,
                        [geom],
                        crop=True,
                        filled=True,
                        nodata=np.nan,
                        indexes=args.band,
                    )
                except ValueError:
                    rows.append({"feature_id": str(fid), "error": "no overlap"})
                    continue
                flat = arr.astype(np.float64).ravel()
                valid = np.isfinite(flat)
                if valid.sum() < 1:
                    rows.append({"feature_id": str(fid), "n_pixels": 0, "mean": "", "median": "", "std": ""})
                    continue
                v = flat[valid]
                rows.append(
                    {
                        "feature_id": str(fid),
                        "n_pixels": int(valid.sum()),
                        "mean": float(np.mean(v)),
                        "median": float(np.median(v)),
                        "std": float(np.std(v)),
                    }
                )

    if not rows:
        raise SystemExit("No features processed")

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
