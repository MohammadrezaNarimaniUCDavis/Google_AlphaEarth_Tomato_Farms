# Alpha Earth notebooks

| Notebook | Purpose |
|----------|---------|
| [`01_clip_alpha_earth.ipynb`](01_clip_alpha_earth.ipynb) | Clip or export embeddings to tomato polygons across configured years (after rasters exist and `load_raster_for_year` is implemented). |

**Prerequisite:** Tomato GPKG from [`../landiq/years/2016/02_filter_tomato_polygons.ipynb`](../landiq/years/2016/02_filter_tomato_polygons.ipynb) or `python -m src.landiq.filter_tomato` (path from `landiq_tomato_gpkg_path` in config, e.g. `landiq_tomato_2016.gpkg`).

Run Jupyter from the repo root with `PYTHONPATH` set (see root [`README.md`](../../README.md)).
