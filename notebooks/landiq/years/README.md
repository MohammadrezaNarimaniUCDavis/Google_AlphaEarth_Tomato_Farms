# LandIQ notebooks by survey year

Aligned with `data/raw/landiq/<YEAR>/` for **2018, 2019, 2020, 2021, 2022, 2023, 2024**.

Only **numeric year folders** appear here. Anything that applies to **every** year lives under **[`shared/`](shared/README.md)**.

| Location | Contents |
|----------|----------|
| **[`shared/`](shared/)** | [`explore_landiq_year.ipynb`](shared/explore_landiq_year.ipynb) — set `YEAR`, run for any listed vintage |
| [`2018/`](2018/) | Reference tutorials: [`01_2018_explore…`](2018/01_2018_explore_shapefile_and_tomato_codes.ipynb) + [`02_filter…`](2018/02_filter_tomato_polygons.ipynb) |
| [`2019/`](2019/) … [`2024/`](2024/) | Per-year README; use **2018** filter notebook + `landiq.year` in config |

**Filter notebook (after `paths.local.yaml`):** [`2018/02_filter_tomato_polygons.ipynb`](2018/02_filter_tomato_polygons.ipynb) — set `landiq.year` and paths for each survey year.

**Config snippets:** [`../../../configs/landiq/years/`](../../../configs/landiq/years/)
