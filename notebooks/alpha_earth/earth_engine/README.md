# Alpha Earth via Google Earth Engine

**By survey year:** [`years/README.md`](years/README.md) — e.g. **2018** pilot [`years/2018/01_pilot_landiq2018_alphaearth_ee.ipynb`](years/2018/01_pilot_landiq2018_alphaearth_ee.ipynb).

Each run downloads **64 embedding bands** (`A00`–`A63`) from [`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) for **every** tomato polygon in the GPKG when `alpha_earth.gee.pilot_polygon_count` is **`null`**, or the **first *N*** when set to an integer (default **5** if the key is omitted). The notebook builds **one** annual mosaic (no per-polygon collection `getInfo`) and uses **threads** so each worker runs **EE download + local polygon→NaN mask** in the same thread (`download_workers`, default **8**). Keep **`scale_m: 10`** for native catalog resolution. Outputs: `data/derived/alpha_earth_clips/ee/landiq<YEAR>/<run_id>_…/` + `manifest.json` (gitignored).

**References:** [leafmap AlphaEarth](https://leafmap.org/maplibre/AlphaEarth/), [EE embedding tutorial](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction).

**Prerequisite:** Tomato GeoPackage from [`../../landiq/years/2018/02_filter_tomato_polygons.ipynb`](../../landiq/years/2018/02_filter_tomato_polygons.ipynb) (or your survey-year folder) and `configs/paths.local.yaml` with matching `landiq.year`.

Run Jupyter from the **repository root** with `pip install -e .` (see root [`README.md`](../../../README.md)).
