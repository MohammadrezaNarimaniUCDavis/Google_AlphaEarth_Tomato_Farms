# Modeling (AlphaEarth → tomato vs non-tomato)

**Start here for workflow, SageMaker, and S3:** **[`guide/README.md`](../guide/README.md)**

## What this is

- **Pixelwise** U-Net on multi-band AlphaEarth embedding **GeoTIFF chips** (resized to `target_hw` in config).
- **Chip-level label**: whole file tomato or non-tomato; **valid** pixels only; **NaNs** masked.
- **Loss**: masked BCE + soft Dice; **metrics**: pixel accuracy, precision, recall, IoU.
- **Uncertainty (inference)**: MC Dropout in `src/modeling/infer_mc.py`.

## Layout

| Path | Role |
|------|------|
| `configs/modeling/tomato_unet.yaml` | Default training config |
| `src/modeling/` | Dataset, U-Net, losses, metrics, training loop |
| `modeling/train/train.py` | CLI entry (`python modeling/train/train.py --config ...`) |
| `outputs/experiments/<run_id>/` | Metrics, checkpoints (gitignored) |
| `data/splits/chips_index.csv` | From `tools/build_chips_index.py` |

## Commands (see `guide/03-data-s3-and-training.md` for Studio vs local)

```bash
export PYTHONPATH=.
python tools/build_chips_index.py
python modeling/train/train.py --config configs/modeling/tomato_unet.yaml
```

## Inference

`modeling/inference/` — placeholder for batch/real-time code when needed.
