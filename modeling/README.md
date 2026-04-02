# Modeling (AlphaEarth → tomato vs non-tomato)

**Start here for workflow, SageMaker, and S3:** **[`guide/README.md`](../guide/README.md)**

## What this is

- **Pixelwise** U-Net on multi-band AlphaEarth embedding **GeoTIFF chips** (resized to `target_hw` in config).
- **Chip-level label**: whole file tomato or non-tomato; **valid** pixels only; **NaNs** masked.
- **Loss**: masked BCE + soft Dice; **metrics**: micro-averaged pixel accuracy, precision, recall, F1, IoU; **chip_acc** (mean-probability vs chip label); **confusion** JSONs per split (`confusion_*_last.json`, `confusion_test.json`).
- **Uncertainty (inference)**: MC Dropout in `src/modeling/infer_mc.py`.

## Layout

| Path | Role |
|------|------|
| `configs/modeling/tomato_unet.yaml` | Default full training config |
| `configs/modeling/tomato_unet_smoke.yaml` | Fast S3+GPU check (`max_train_batches` / `max_eval_batches`) |
| `src/modeling/` | Dataset, U-Net, losses, metrics, training loop |
| `modeling/train/train.py` | CLI entry (`python modeling/train/train.py --config ...`) |
| `outputs/experiments/<run_id>/` | Metrics, checkpoints (gitignored) |
| `data/splits/chips_index.csv` | From `tools/build_chips_index.py` |

## Commands (see `guide/03-data-s3-and-training.md` for Studio vs local)

```bash
export PYTHONPATH=.
python tools/build_chips_index.py
# Quick check (~few minutes from S3):
python modeling/train/train.py --config configs/modeling/tomato_unet_smoke.yaml
# Or: python modeling/train/train.py --config configs/modeling/tomato_unet.yaml --smoke
# Full run:
python modeling/train/train.py --config configs/modeling/tomato_unet.yaml
```

## Inference

`modeling/inference/` — placeholder for batch/real-time code when needed.
