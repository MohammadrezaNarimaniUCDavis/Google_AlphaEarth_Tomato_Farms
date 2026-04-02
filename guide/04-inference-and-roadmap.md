# Inference, uncertainty, zonal stats — full pipeline

## Speed reminder

Mirror chips once, then train/infer from disk:

```bash
./tools/sync_alphaearth_clips_from_s3.sh
export ALPHA_EARTH_DATA_SOURCE=auto
```

## 1. One chip (`infer_chip.py`)

```bash
python modeling/inference/infer_chip.py \
  --checkpoint outputs/experiments/<run_id>/best.pt \
  --row-index 0 --split val \
  --mc-samples 20 \
  --geotiff
```

- **NPZ** + **`aggregate.json`** always (under `outputs/predictions/<run_id>/`).
- **`--geotiff`**: writes **`pred_mean_prob.tif`** (and **`pred_var_prob.tif`** if MC) **only if the source `.tif` exists on local disk** (after sync). From S3-only paths, skip sync or omit `--geotiff`.

## 2. Many chips (`infer_batch.py`)

```bash
python modeling/inference/infer_batch.py \
  --checkpoint outputs/experiments/<run_id>/best.pt \
  --split test --limit 100 \
  --mc-samples 10 \
  --geotiff
```

- One **subfolder per `chip_id`** under `outputs/predictions/batch_<run>_<split>/`.
- **`batch_summary.json`**: list of per-chip `aggregate` (and errors).

## 3. Large GeoTIFF — sliding window (`infer_tile.py`)

Same **band count** as training chips. Patches are read from the big raster, **resized to `target_hw`**, inferred, then **blended** back with overlap-weighted average.

```bash
python modeling/inference/infer_tile.py \
  --checkpoint outputs/experiments/<run_id>/best.pt \
  --input path/to/large_embedding.tif \
  --overlap 32 \
  --out outputs/predictions/region_mean_prob.tif \
  --mc-samples 0
```

Optional **`--out-var`** when **`--mc-samples` > 0**.

## 4. Multi-farm / polygon zonal stats (`zonal_stats.py`)

Requires **`pip install fiona`** (listed in `requirements-modeling*.txt`).

```bash
python modeling/inference/zonal_stats.py \
  --raster outputs/predictions/region_mean_prob.tif \
  --vector path/to/farms.gpkg \
  --id-field farm_id \
  --out-csv outputs/predictions/zonal_by_farm.csv
```

Per polygon: **n_pixels**, **mean**, **median**, **std** of raster values (tomato probability).

## 5. Archive runs to S3 (manual)

```bash
aws s3 sync outputs/experiments/ s3://tomato-alphaearth-054037103012-data/google-alphaearth-tomato-farms/experiments/
aws s3 sync outputs/predictions/ s3://tomato-alphaearth-054037103012-data/google-alphaearth-tomato-farms/predictions/
```

(Requires **PutObject** on that prefix — see `tools/aws-preflight/sagemaker-execution-s3-tomato-bucket-policy.json`.)

## Still optional (paper / polish)

- **TensorBoard / W&B** — not wired in `train.py`.
- **RF baseline** on embedding vectors (Earth Engine style) — separate script or notebook.
- **Calibration** (temperature scaling), **DeepLab** backbone, **deep ensemble** — future extensions.

## GitHub

Push from a machine with credentials:

```bash
git push origin main
```

Studio often has no stored GitHub token; use laptop CI or PAT on Studio if needed.
