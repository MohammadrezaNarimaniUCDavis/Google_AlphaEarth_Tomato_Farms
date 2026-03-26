# Alpha Earth notebooks

| Notebook | Purpose |
|----------|---------|
| [`earth_engine/years/2018/01_pilot_landiq2018_alphaearth_ee.ipynb`](earth_engine/years/2018/01_pilot_landiq2018_alphaearth_ee.ipynb) | **Earth Engine pilot (2018):** download **64-band** embeddings for the first *N* tomato polygons. |
| [`years/2018/01_clip_local_alphaearth_2018_tomato.ipynb`](years/2018/01_clip_local_alphaearth_2018_tomato.ipynb) | **Local GeoTIFF:** clip a **2018** stack to the tomato polygon union from LandIQ filtering. |
| [`years/README.md`](years/README.md) | Index — local clips by survey year. |
| [`earth_engine/years/README.md`](earth_engine/years/README.md) | Index — EE pilots by survey year. |
| [`01_clip_alpha_earth.ipynb`](01_clip_alpha_earth.ipynb) | Short pointer to **`years/`** (no code here). |

**Earth Engine:** [`earth_engine/README.md`](earth_engine/README.md) — Satellite Embedding V1 ([GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)). Optional [leafmap AlphaEarth](https://leafmap.org/maplibre/AlphaEarth/).

**Prerequisite:** Tomato GPKG from [`../landiq/years/2018/02_filter_tomato_polygons.ipynb`](../landiq/years/2018/02_filter_tomato_polygons.ipynb) or `python -m src.landiq.filter_tomato` (default `landiq_tomato_<year>.gpkg`).

Run Jupyter from the repo root with `pip install -e .` (see root [`README.md`](../../README.md)).
