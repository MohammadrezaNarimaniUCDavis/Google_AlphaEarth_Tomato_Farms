# Data, S3, and training on Studio

## S3 bucket (this project)

- **Bucket:** `tomato-alphaearth-054037103012-data`
- **Region:** `us-west-2`
- **Prefix:** `google-alphaearth-tomato-farms/`

Layout (see `data/s3/README.md`):

| Content | Key prefix under `google-alphaearth-tomato-farms/` |
|---------|-----------------------------------------------------|
| AlphaEarth clips | `derived/alpha_earth_clips/` |
| Splits / index | `splits/chips_index.csv` |
| Experiments (optional mirror) | `experiments/` |

## Repo config

- **`configs/paths.example.yaml`** — documents `s3:` (copy to **`configs/paths.local.yaml`** on a machine with secrets; `paths.local` is gitignored).
- For **committed** `chips_index.csv` with `s3_uri` columns, generate on a machine where `paths.local.yaml` includes `s3.bucket` (or merge example into local and run `build_chips_index`).

## Build chip index (local or Studio with full `data/derived/...`)

```bash
export PYTHONPATH=.   # Windows PowerShell: $env:PYTHONPATH="."
python tools/build_chips_index.py
```

Writes `data/splits/chips_index.csv` (+ parquet if pyarrow available).

## Studio: clone repo and use S3-only chips

```bash
git clone https://github.com/<org>/Google_AlphaEarth_Tomato_Farms.git
cd Google_AlphaEarth_Tomato_Farms
```

If `chips_index.csv` is already in the repo with `s3_uri`, or you copy from S3:

```bash
mkdir -p data/splits
aws s3 cp s3://tomato-alphaearth-054037103012-data/google-alphaearth-tomato-farms/splits/chips_index.csv data/splits/
```

**Read GeoTIFFs from S3** (no local mirror):

```bash
export ALPHA_EARTH_DATA_SOURCE=s3
```

Requires Studio execution role (or your credentials) with **S3 read** on the bucket.

## Install dependencies on Studio (GPU)

PyTorch **2.2.x** + CUDA **cu121** wheels (adjust if Studio image provides PyTorch already):

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-modeling-cpu.txt
pip install -e . --no-deps
```

## Train

```bash
python modeling/train/train.py --config configs/modeling/tomato_unet.yaml
```

Outputs under `outputs/experiments/<run_id>/` (see `.gitignore`; copy to S3 to archive):

| File | Contents |
|------|-----------|
| `metrics_epoch.csv` | Per epoch: `train_loss_opt` (minibatch mean), train/val pixel `acc`, `precision`, `recall`, `f1`, `iou`, `chip_acc`, val loss |
| `confusion_train_last.json` / `confusion_val_last.json` | Micro-averaged 2×2 confusion (updated each epoch) |
| `confusion_test.json` | Test split confusion after best checkpoint |
| `metrics_test.json` | Test pixel + chip metrics |
| `best.pt` / `last.pt` | Weights + config snapshot |

## Local Windows (`gee` conda) — CPU only

```powershell
conda activate gee
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-modeling-cpu.txt
pip install -e . --no-deps
```

Use only for **imports / tiny** forward pass; full training on Studio.

## Sync scripts (from Windows repo root)

```powershell
.\tools\aws-preflight\sync-alphaearth-clips-to-s3.ps1
.\tools\aws-preflight\sync-splits-to-s3.ps1
```
