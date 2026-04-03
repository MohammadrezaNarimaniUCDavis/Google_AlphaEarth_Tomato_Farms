# SageMaker Studio snapshot (this folder is tracked in Git)

This directory is a **point-in-time copy** of training **outputs** from **SageMaker Studio** (user home: `/home/sagemaker-user`), taken so metrics, checkpoints, and logs survive after the remote environment is shut down.

## What is included

| Path | Contents |
|------|----------|
| **`project-outputs/experiments/`** | All experiment run folders: `metrics_epoch.csv`, `metrics_test.json`, `confusion_*.json`, `config_resolved.json`, `best.pt` / `last.pt`, `experiment_complete.json`, resume checkpoints, `s3_sync.log`, etc. |
| **`project-outputs/predictions/`** | Sample inference outputs (e.g. `batch_*`, NPZ + `aggregate.json`). |
| **`project-outputs/train_full.log`** | Full training log (terminal capture). |
| **`project-outputs/train_latest.log`** | Latest short log. |
| **`project-outputs/*.bak*`** | Rotated log fragments if present. |
| **`HOME_DIRECTORY_LISTING.txt`** | Non-hidden listing of `/home/sagemaker-user` at snapshot time (names only). |
| **`SNAPSHOT_UTC.txt`** | UTC timestamp + `uname` line for the Studio instance. |

## What is **not** included (by design)

- **Secrets:** `.bash_history`, AWS keys, SSH keys, `.env`, GEE credentials — never commit these.
- **Huge mirrors:** `data/derived/alpha_earth_clips/` (use S3 sync scripts to recover).
- **IDE / server metadata:** `.cursor/`, `sagemaker-code-editor-server-data/` (large, machine-specific).
- **Whole-disk home backup:** only **this project’s `outputs/`** and a **home directory listing** are archived here.

Canonical long-term storage for experiments is also **S3** (see `project-outputs/experiments/<run_id>/s3_sync.log` and project docs under `guide/`).

## Using checkpoints from this archive

Point inference scripts at a checkpoint under:

`archive/sagemaker-studio/project-outputs/experiments/<run_id>/best.pt`

Example:

```bash
python modeling/inference/infer_chip.py \
  --checkpoint archive/sagemaker-studio/project-outputs/experiments/20260402T230522Z/best.pt \
  --row-index 0 --split val
```

## Size note

Checkpoints (`.pt`) are ~31 MB each; full experiment folders are tens of MB. This is intentional so GitHub holds a **recoverable** record without relying only on S3.
