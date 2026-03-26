# Alpha Earth — local raster clips by LandIQ survey year

Place **exported GeoTIFF / COG** stacks under `data/derived/alpha_earth_rasters/<YEAR>/` (see `alpha_earth.local_raster_root` in `configs/paths.example.yaml`). Tomato polygons come from LandIQ filtering (`landiq_tomato_<year>.gpkg`).

| Survey year | Notebook |
|-------------|----------|
| **2018** | [`2018/01_clip_local_alphaearth_2018_tomato.ipynb`](2018/01_clip_local_alphaearth_2018_tomato.ipynb) |

**Earth Engine** (download embeddings as GeoTIFF per polygon): [`../earth_engine/years/README.md`](../earth_engine/years/README.md).
