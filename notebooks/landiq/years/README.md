# LandIQ notebooks by survey year

Aligned with `data/raw/landiq/<YEAR>/` for **2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024**.

Only **numeric year folders** appear here. Anything that applies to **every** year lives under **[`shared/`](shared/README.md)**.

| Location | Contents |
|----------|----------|
| **[`shared/`](shared/)** | [`explore_landiq_year.ipynb`](shared/explore_landiq_year.ipynb) — set `YEAR`, run for any listed vintage |
| [`2016/`](2016/) | [`01_2016_explore…`](2016/01_2016_explore_shapefile_and_tomato_codes.ipynb) + [`02_filter…`](2016/02_filter_tomato_polygons.ipynb) |
| [`2018/`](2018/) … [`2024/`](2024/) | Per-year README |

**Filter notebook (after `paths.local.yaml`):** [`2016/02_filter_tomato_polygons.ipynb`](2016/02_filter_tomato_polygons.ipynb) — set `landiq.year` in config for each survey year.

**Config snippets:** [`../../../configs/landiq/years/`](../../../configs/landiq/years/)
