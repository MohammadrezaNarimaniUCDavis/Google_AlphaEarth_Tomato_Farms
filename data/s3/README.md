# S3 data layout (SageMaker + this project)

Large artifacts should live in **Amazon S3** so SageMaker (or any machine) can read them without copying the whole dataset onto the notebook disk.

## Bucket (from our CloudFormation stack)

- **Name pattern:** `tomato-alphaearth-<AWS_ACCOUNT_ID>-data`
- **Example (digitalaglab):** `s3://tomato-alphaearth-054037103012-data/`
- **Region:** `us-west-2` (match your SageMaker notebook region)

Create the bucket with `infra/sagemaker-notebook.yaml` or reuse another bucket by setting `s3.bucket` in `configs/paths.local.yaml`.

**New buckets are empty.** Large exports (GeoTIFFs) stay on your machine until you upload them — there is no automatic copy from your laptop to S3. Use the sync script below.

## Designated “project folder” inside the bucket

All keys for this repo live under **one prefix** so nothing collides with other projects in the same bucket:

```text
google-alphaearth-tomato-farms/
```

Full URI pattern:

```text
s3://tomato-alphaearth-<ACCOUNT_ID>-data/google-alphaearth-tomato-farms/<category>/...
```

Mirror your **local** `data/` layout under that prefix so paths are predictable:

| Local (repo) | S3 key prefix (under `google-alphaearth-tomato-farms/`) |
|--------------|---------------------------------------------------------|
| `data/raw/landiq/<year>/` | `raw/landiq/<year>/` |
| `data/derived/landiq_tomato/` | `derived/landiq_tomato/` |
| `data/derived/landiq_non_tomato/` | `derived/landiq_non_tomato/` |
| `data/derived/alpha_earth_clips/` (EE exports) | `derived/alpha_earth_clips/` e.g. `ee/landiq2018/<run_id>_…/` |
| manifests, train/val lists | `manifests/` |
| model checkpoints | `models/` |
| `data/splits/` | `splits/` |
| training runs (metrics, optional mirror of `outputs/experiments/`) | `experiments/` |

## Upload / sync AlphaEarth clips (tomato + non-tomato TIFFs)

From the **repo root**, with GeoTIFFs under `data/derived/alpha_earth_clips/`:

```powershell
.\tools\aws-preflight\sync-alphaearth-clips-to-s3.ps1
```

Dry-run only:

```powershell
.\tools\aws-preflight\sync-alphaearth-clips-to-s3.ps1 -DryRun
```

Override bucket/profile if needed (edit the script’s defaults or extend with parameters). First sync of **~10k** files can take **a long time** depending on uplink size.

## Config in this repo

- **`configs/paths.example.yaml`** → section **`s3:`** (`bucket`, `project_root_prefix`, `keys`).
- **`src/utils/s3_layout.py`** → build `s3://…` URIs and key prefixes for notebooks/scripts.
