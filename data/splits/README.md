# Splits index for AlphaEarth chips

This folder is where we store the **train/val/test index** for the tomato vs
non_tomato AlphaEarth chips.

It is **not** committed to git (see `.gitignore`) – you generate it locally
from the clipped GeoTIFFs.\n
\n
## How to build\n
\n
From the repo root:\n
\n
```bash\n
python tools/build_chips_index.py\n
```\n
\n
This scans `data/derived/alpha_earth_clips/ee/...` for `.tif`/`.tiff` files,\nassigns labels based on folder names (`landiq2018` → tomato,\n`landiq2018_non_tomato` → non_tomato), balances the two classes, and writes:\n\n- `chips_index.parquet` (if Parquet support is installed) and\n- `chips_index.csv`\n\n### Columns\n\n- `chip_index` – numeric index into the internal file list (stable order)\n- `chip_id` – filename stem (no extension)\n- `class_label` – `tomato` or `non_tomato`\n- `split` – `train`, `val`, or `test`\n- `local_path` – repo-relative path to the GeoTIFF\n- `s3_uri` – S3 URI if `s3.bucket` is configured in `configs/paths.local.yaml`\n\nDownstream notebooks (including the SageMaker training notebook) should\nconsume this index instead of re-scanning the filesystem.\n+
