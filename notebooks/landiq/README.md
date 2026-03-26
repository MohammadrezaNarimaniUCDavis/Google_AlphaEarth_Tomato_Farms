# LandIQ notebooks

## By survey year (`years/`)

Mirrors `data/raw/landiq/<YEAR>/` for **2018–2024**. See **[`years/README.md`](years/README.md)**.

- **[`years/shared/explore_landiq_year.ipynb`](years/shared/explore_landiq_year.ipynb)** — set `YEAR`, run for any listed vintage  
- **[`years/2018/`](years/2018/)** — reference **`01_…explore…`** + **`02_filter_tomato_polygons.ipynb`**  
- **`years/2019/` … `years/2024/`** — per-year READMEs; filter uses the **2018** notebook with `landiq.year` updated in config  

**Order:** explore under **`years/<YEAR>/`** (template: **2018**) → edit **`configs/paths.local.yaml`** → **[`years/2018/02_filter_tomato_polygons.ipynb`](years/2018/02_filter_tomato_polygons.ipynb)** (or `python -m src.landiq.filter_tomato`).

Run from repo root with `pip install -e .` in your env (see root [`README.md`](../../README.md)).
