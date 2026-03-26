# LandIQ config snippets by survey year

Copy fields into `configs/paths.local.yaml` under `landiq:`. Files are **not** auto-loaded.

| Year | Example snippet |
|------|-----------------|
| 2018 | [`2018.example.yaml`](2018.example.yaml) |
| 2019 | [`2019.example.yaml`](2019.example.yaml) |
| 2020 | [`2020.example.yaml`](2020.example.yaml) |
| 2021 | [`2021.example.yaml`](2021.example.yaml) |
| 2022 | [`2022.example.yaml`](2022.example.yaml) |
| 2023 | [`2023.example.yaml`](2023.example.yaml) |
| 2024 | [`2024.example.yaml`](2024.example.yaml) |

**Years in code:** `SURVEY_YEARS` in [`src/landiq/years/registry.py`](../../../src/landiq/years/registry.py). Update `ZIP_FILENAME_BY_YEAR` when you confirm each zip basename under `data/raw/landiq/<YEAR>/`.
