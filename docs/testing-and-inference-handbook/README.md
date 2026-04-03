# Testing & inference handbook

Hands-on guide for **training checks**, **running inference at different scales**, and **interpreting outputs** (tomato vs non-tomato, per pixel vs aggregated). For **canonical commands** and S3 sync, see also [`guide/04-inference-and-roadmap.md`](../../guide/04-inference-and-roadmap.md) and [`guide/03-data-s3-and-training.md`](../../guide/03-data-s3-and-training.md).

**Assumptions:** repo root is `Google_AlphaEarth_Tomato_Farms`, shell is run from that directory unless noted.

---

## 1. What this model is (in one minute)

- **Architecture:** U-Net on **multi-band AlphaEarth embedding GeoTIFFs** (chips resized to `target_hw` from the training config).
- **Task:** **Binary pixelwise segmentation** — each pixel gets a **probability** of belonging to the “tomato” class. The training labels are **chip-level** (whole chip tomato or not), applied to valid pixels; invalid pixels are masked.
- **Not:** A single “this image is tomato / not tomato” classifier. You can **derive** a coarse score per chip or region by **aggregating** pixel probabilities (mean, fraction above threshold, zonal stats).

Use **`best.pt`** from an experiment folder for inference unless you have a reason to use **`last.pt`**.

---

## 2. Training — quick reference (for reproducibility and smoke tests)

### 2.1 Environment

```bash
cd Google_AlphaEarth_Tomato_Farms
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121   # adjust if image has PyTorch
pip install -r requirements-modeling-cpu.txt
pip install -e . --no-deps
export PYTHONPATH=.
```

Data access (pick one workflow):

| Goal | Command / note |
|------|----------------|
| Read chips from S3 only | `export ALPHA_EARTH_DATA_SOURCE=s3` |
| Prefer local mirror if present | `./tools/sync_alphaearth_clips_from_s3.sh` then `export ALPHA_EARTH_DATA_SOURCE=auto` |

### 2.2 Smoke vs full training

| Run | Config | Purpose |
|-----|--------|---------|
| **Smoke** (~few minutes) | `configs/modeling/tomato_unet_smoke.yaml` or `tomato_unet.yaml --smoke` | Pipeline OK, GPU/S3, code paths |
| **Full** | `configs/modeling/tomato_unet.yaml` | Real metrics; no `max_train_batches` / `max_eval_batches` limits |

```bash
# Smoke
python modeling/train/train.py --config configs/modeling/tomato_unet_smoke.yaml

# Full
python modeling/train/train.py --config configs/modeling/tomato_unet.yaml
```

### 2.3 Where training writes artifacts

Under **`outputs/experiments/<run_id>/`** (gitignored locally; often synced to S3):

| File | Meaning |
|------|---------|
| `best.pt` / `last.pt` | Checkpoints (weights + config snapshot) — **use `best.pt` for inference** |
| `metrics_epoch.csv` | Per-epoch train/val metrics |
| `metrics_test.json`, `confusion_test.json` | Test split after training |
| `experiment_complete.json` | Final summary when the run finished cleanly |

Replace `<run_id>` below with your folder name (example: `20260402T230522Z`).

---

## 3. What the model outputs (by “level”)

This pipeline is **pixel-wise** segmentation. Outputs depend on **how** you run inference.

| Level | What you get | How to think about it |
|-------|----------------|------------------------|
| **Per pixel** | Probability **p**(tomato) in **\[0, 1\]**, same **H×W** as the model input (after resize to `target_hw`). | Threshold (e.g. 0.5) → **binary mask** tomato / not tomato **per pixel**. |
| **Per chip** | **`aggregate.json`** (and batch summaries) with **summaries over valid pixels** on that chip. | Not a single hard-coded “tomato” label — see **§4** for field names. |
| **Optional uncertainty** | MC Dropout: **mean** and **variance** of **p** across stochastic forward passes. | `--mc-samples N` with `N > 0`; GeoTIFF: **`pred_mean_prob.tif`**, **`pred_var_prob.tif`** when `--geotiff` and source `.tif` is **local**. |
| **Per region / farm** | CSV: **n_pixels**, **mean**, **median**, **std** of the **probability raster** inside each polygon. | Run **`zonal_stats.py`** on a probability GeoTIFF + vector layer. This is **statistics of p**, not a built-in “farm class” output. |

**Entry points (scripts live under `modeling/inference/`):**

| Script | Use case |
|--------|----------|
| `infer_chip.py` | One chip (from index row or `--local-path`) |
| `infer_batch.py` | Many chips from a split (`chips_index`) |
| `infer_tile.py` | Large embedding GeoTIFF → **blended** probability raster |
| `zonal_stats.py` | Polygons → per-feature stats from a raster |

---

## 4. `aggregate.json` (per-chip summaries)

Written next to **`mean_prob.npz`**. The **`aggregate`** object is produced by `chip_aggregate` in code; typical fields:

| Field | Meaning |
|-------|---------|
| `mean_p_tomato` | Mean of **p**(tomato) over **valid** pixels |
| `median_p_tomato` | Median of **p** over valid pixels |
| `frac_valid` | Fraction of pixels that are valid (mask) |
| `mean_uncertainty` | If MC variance used: mean uncertainty over valid pixels |
| `frac_high_uncertainty` | If MC used: fraction of pixels with uncertainty above a quantile threshold |

The file also includes **`chip_id`** and metadata (checkpoint path, etc.) merged from the inference run.

---

## 5. Step-by-step: test inference at each level

Use a real checkpoint path:

```bash
export RUN_ID=20260402T230522Z   # change to your experiment id
export CKPT=outputs/experiments/${RUN_ID}/best.pt
```

### 5.1 Level A — one chip (sanity check)

Fastest way to verify the model loads and produces arrays + JSON.

```bash
python modeling/inference/infer_chip.py \
  --checkpoint "${CKPT}" \
  --row-index 0 --split val \
  --mc-samples 0
```

**Expected:** directory under **`outputs/predictions/<run_id>/`** (infer script chooses a run id; check terminal output) with **`mean_prob.npz`**, **`aggregate.json`**.

**Optional — probability GeoTIFF (needs the source chip `.tif` on disk):**

```bash
python modeling/inference/infer_chip.py \
  --checkpoint "${CKPT}" \
  --row-index 0 --split val \
  --mc-samples 20 \
  --geotiff
```

- **`--mc-samples 20`:** also writes **`var_prob.npz`** and, with `--geotiff`, **`pred_var_prob.tif`** if the source file exists locally.
- If the chip is **S3-only** and not mirrored, **`--geotiff`** may skip GeoTIFF export — sync clips first (`./tools/sync_alphaearth_clips_from_s3.sh`) or use **`--local-path`** to a known file.

**Custom file:**

```bash
python modeling/inference/infer_chip.py \
  --checkpoint "${CKPT}" \
  --local-path data/derived/alpha_earth_clips/ee/.../your_chip.tif \
  --mc-samples 0 \
  --geotiff
```

**Inspect:** open **`aggregate.json`**; load **`mean_prob.npz`** in Python (`np.load`) and check **`prob`** shape matches `target_hw`.

### 5.2 Level B — many chips (batch evaluation / QA)

```bash
python modeling/inference/infer_batch.py \
  --checkpoint "${CKPT}" \
  --split test \
  --limit 10 \
  --mc-samples 0 \
  --geotiff
```

**Expected:** **`outputs/predictions/batch_<run>_<split>/`** with **one subfolder per `chip_id`**, plus **`batch_summary.json`** listing per-chip aggregates (and any errors).

Increase **`--limit`** for broader coverage; use **`--split val`** or **`test`** as needed.

### 5.3 Level C — large GeoTIFF (full scene / region)

Same **band count** as training chips. Patches are read in a sliding window, resized to `target_hw`, inferred, then **overlap-blended** into one raster.

```bash
python modeling/inference/infer_tile.py \
  --checkpoint "${CKPT}" \
  --input path/to/large_embedding.tif \
  --overlap 32 \
  --out outputs/predictions/region_mean_prob.tif \
  --mc-samples 0
```

**With MC uncertainty** (also pass **`--out-var`** when `mc-samples > 0` — see **`infer_tile.py --help`**):

```bash
python modeling/inference/infer_tile.py \
  --checkpoint "${CKPT}" \
  --input path/to/large_embedding.tif \
  --overlap 32 \
  --out outputs/predictions/region_mean_prob.tif \
  --out-var outputs/predictions/region_var_prob.tif \
  --mc-samples 10
```

**Inspect:** open **`region_mean_prob.tif`** in QGIS; values are **tomato probability** per pixel.

### 5.4 Level D — per farm / polygon (zonal aggregation)

Requires **`fiona`** (see `requirements-modeling*.txt`).

```bash
pip install fiona   # if not already installed

python modeling/inference/zonal_stats.py \
  --raster outputs/predictions/region_mean_prob.tif \
  --vector path/to/farms.gpkg \
  --id-field farm_id \
  --out-csv outputs/predictions/zonal_by_farm.csv
```

**Expected:** CSV with per polygon: **n_pixels**, **mean**, **median**, **std** of raster values (i.e. of **p**(tomato)). Interpret **mean** as “average predicted tomato probability inside this boundary,” not a calibrated farm-level class probability unless you define a separate rule.

---

## 6. Optional: archive experiments / predictions to S3

Requires AWS CLI credentials with **PutObject** on your bucket prefix (see **`tools/aws-preflight/sagemaker-execution-s3-tomato-bucket-policy.json`**).

```bash
aws s3 sync outputs/experiments/ s3://tomato-alphaearth-054037103012-data/google-alphaearth-tomato-farms/experiments/
aws s3 sync outputs/predictions/ s3://tomato-alphaearth-054037103012-data/google-alphaearth-tomato-farms/predictions/
```

Adjust bucket/prefix if your project uses a different location.

---

## 7. Related docs in this repo

| Document | Topics |
|----------|--------|
| [`guide/04-inference-and-roadmap.md`](../../guide/04-inference-and-roadmap.md) | Inference commands, MC, tiling, zonal stats, S3 sync |
| [`guide/03-data-s3-and-training.md`](../../guide/03-data-s3-and-training.md) | S3 layout, `chips_index`, training, metrics caveats |
| [`modeling/README.md`](../../modeling/README.md) | Train/infer entrypoints overview |
| [`modeling/inference/README.md`](../../modeling/inference/README.md) | Short CLI table |

---

## 8. Troubleshooting (short)

| Issue | What to check |
|-------|----------------|
| `best.pt` not found | Training finished? Path uses correct **`run_id`**? |
| GeoTIFF not written | Source `.tif` must exist **locally** for `--geotiff`; mirror from S3 first. |
| Shape / band errors | Input band count must match training; **`infer_tile`** uses same resize as training. |
| “Too good” smoke metrics | Smoke runs use **few batches** — not representative; see **`guide/03`** metrics caveats. |

For SageMaker Studio setup (GPU, IAM, Cursor), see **`guide/02-sagemaker-cursor-remote.md`**.
