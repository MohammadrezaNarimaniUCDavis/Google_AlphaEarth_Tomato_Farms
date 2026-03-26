# Data layout

## `raw/`

**LandIQ** lives under `raw/landiq/`. This directory is **gitignored** because files are large and often license-restricted.

Typical layout (by survey year — **2018–2024** in this project):

```text
raw/landiq/
  2018/
  2019/
  2020/
  2021/
  2022/
  2023/
  2024/
    *_Legend.pdf
    *_crop_mapping_*_provisional.zip   # extract here (or into a subfolder)
```

**Extract each ZIP** in the matching year folder so that `.shp` (and `.dbf`, `.shx`, `.prj`, …) are on disk. GeoPandas cannot read shapefiles still inside the archive.

Notebooks under `notebooks/landiq/years/<YEAR>/` mirror `data/raw/landiq/<YEAR>/`. Extracted copies go to `data/derived/landiq_extracted/<year>/` (gitignored). Per-year config snippets: `configs/landiq/years/*.example.yaml`.

Do not commit raw imagery or Alpha Earth exports unless your license allows it and file sizes are manageable.

## `derived/`

| Subfolder | Contents |
|-----------|----------|
| `landiq_tomato/` | Tomato-only polygons (e.g. GeoPackage or GeoJSON) produced after filtering. |
| `alpha_earth_rasters/` | Local GeoTIFF / COG stacks, typically `<year>/*.tif` (see `alpha_earth.local_raster_root`). **Gitignored** when large. |
| `alpha_earth_clips/` | Clipped outputs: `local/landiq<YEAR>/…` or EE `ee/landiq<YEAR>/<run_id>_…/`. **Gitignored** when large. |

## `splits/`

Train, validation, and test definitions for deep learning (e.g. CSV of polygon IDs, GeoJSON subsets, or manifest files). Prefer **spatially** or **temporally** disjoint splits that match your paper’s claims.
