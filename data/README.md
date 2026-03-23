# Data layout

## `raw/`

**LandIQ** lives under `raw/landiq/`. This directory is **gitignored** because files are large and often license-restricted.

Typical layout (by survey year):

```text
raw/landiq/
  2016/
  2018/
  …
  2024/
    *_Legend.pdf
    *_crop_mapping_*_provisional.zip   # extract here (or into a subfolder)
```

**Extract each ZIP** in the matching year folder so that `.shp` (and `.dbf`, `.shx`, `.prj`, …) are on disk. GeoPandas cannot read shapefiles still inside the archive.

Do not commit raw imagery or Alpha Earth exports unless your license allows it and file sizes are manageable.

## `derived/`

| Subfolder | Contents |
|-----------|----------|
| `landiq_tomato/` | Tomato-only polygons (e.g. GeoPackage or GeoJSON) produced after filtering. |
| `alpha_earth_clips/` | Rasters or extracted embeddings clipped to tomato polygons, organized by year. |

## `splits/`

Train, validation, and test definitions for deep learning (e.g. CSV of polygon IDs, GeoJSON subsets, or manifest files). Prefer **spatially** or **temporally** disjoint splits that match your paper’s claims.
