# Local Alpha Earth clip — **2018**

- **Notebook:** [`01_clip_local_alphaearth_2018_tomato.ipynb`](01_clip_local_alphaearth_2018_tomato.ipynb) — clip a **2018** raster stack to the **union** of all tomato polygons from `landiq_tomato_2018.gpkg` (or your `landiq.output_filename`).

**Before you run:** Export or download your **2018** embedding / raster as GeoTIFF and put it under:

`data/derived/alpha_earth_rasters/2018/` (e.g. `embeddings_2018.tif`).

**Prerequisite:** [`../../../landiq/years/2018/02_filter_tomato_polygons.ipynb`](../../../landiq/years/2018/02_filter_tomato_polygons.ipynb).

**Output:** `data/derived/alpha_earth_clips/local/landiq2018/` (gitignored).
