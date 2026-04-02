# Project plan — AlphaEarth tomato vs non-tomato

## One sentence

A **pixelwise** deep model on **AlphaEarth embedding chips** outputs **tomato probability and uncertainty per pixel**, then **aggregates** to chip / polygon / region for metrics and maps—trained in a **reproducible script pipeline** with **data on S3**, **heavy compute on SageMaker GPU**, editing from **Cursor** (local or remote).

## Compared to “all crops + RF in Earth Engine”

Random forest on **sampled** embeddings vs **broad CDL-style** labels is a different problem. This work is **binary tomato vs non-tomato** with **LandIQ-derived** supervision, **full raster** geometry via **segmentation-style** learning, and **explicit uncertainty**—a clearer paper story. RF-on-embeddings can be cited as a **simple baseline** later.

## Data and storage

- **Inputs:** AlphaEarth clips under `data/derived/alpha_earth_clips/ee`, mirrored on S3 (`derived/alpha_earth_clips/`).
- **Supervision:** Whole-chip label → **uniform** per-pixel label on **valid** pixels only; **NaN / NoData** masked out (never trained).
- **Splits:** Train / val / test, **balanced** chip counts per class (`tools/build_chips_index.py`). Optional stratification by year/region when metadata exists.
- **Version:** `data/splits/chips_index.csv` (+ optional parquet), with **`s3_uri`** when `configs/paths.local.yaml` has `s3.bucket` (committed CSV in repo uses S3 URIs for Studio-only workflows).

## Modeling (one coherent system)

| Piece | Role |
|-------|------|
| Encoder + decoder (UNet-style in code) | Per-pixel logits → tomato vs background. |
| Loss | Masked **BCE + Dice** on valid pixels. |
| Uncertainty | **MC Dropout** at inference (`src/modeling/infer_mc.py`): mean *p*(tomato), variance across passes. |
| Aggregation | Per chip/polygon: mean *p*, uncertainty summaries; chip-level decision rules for the paper. |

Ablations: with/without Dice, with/without uncertainty reporting.

## Metrics and logging

- **Per epoch (train/val):** loss; pixel accuracy, IoU, precision, recall, F1 on valid pixels; optional chip-level metrics.
- **Test:** same + confusion matrices (pixel and chip-level).
- **Artifacts:** `outputs/experiments/<run_id>/` — `metrics_epoch.csv`, `metrics_test.json`, `best.pt`, `config_resolved.json`. SageMaker: also `SM_MODEL_DIR` when set.

## Phase 5 (later)

Tiled inference on **large** GeoTIFFs; per-farm zonal stats (“20 farmers” figure).

## Repo layout (implemented)

- `src/modeling/` — dataset, UNet, losses, metrics, training loop.
- `configs/modeling/tomato_unet.yaml` — default training config.
- `modeling/train/train.py` — CLI entry point.

## Workflow split

| Where | What |
|-------|------|
| **Local** | LandIQ / EE prep, plots, optional CPU smoke tests (`conda` env `gee`). |
| **S3** | Clips, splits manifest, optional experiment mirrors. |
| **Studio + GPU** | `train.py`, real training, logs under `outputs/experiments/`. |

SageMaker is a **replaceable compute layer**; code stays in Git.
