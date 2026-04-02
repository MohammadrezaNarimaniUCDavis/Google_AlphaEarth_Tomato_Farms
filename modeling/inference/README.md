# Inference CLIs

| Script | Purpose |
|--------|---------|
| **`infer_chip.py`** | One chip → NPZ + `aggregate.json` + optional GeoTIFF (`--geotiff`, needs local `.tif`). |
| **`infer_batch.py`** | Many chips from `chips_index` split → subfolders + `batch_summary.json`. |
| **`infer_tile.py`** | Large GeoTIFF → blended probability (and optional var) raster. |
| **`zonal_stats.py`** | Polygons (GPKG/GeoJSON/shp) → CSV of mean/median/std per feature. |

Full commands and S3 sync: **`guide/04-inference-and-roadmap.md`**.
