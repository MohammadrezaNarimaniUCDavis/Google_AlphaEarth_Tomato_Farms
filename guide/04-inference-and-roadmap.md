# Inference, uncertainty, and roadmap (full plan)

This doc maps the **scientific plan** to **what exists in the repo** and the **recommended order** to implement the rest.

## Done now (you can run today)

| Piece | Where |
|--------|--------|
| Train U-Net, BCE+Dice, masks, S3 or local | `modeling/train/train.py`, `configs/modeling/tomato_unet.yaml` |
| Per-pixel **probability** (sigmoid of logits) | Training metrics; **inference** below |
| **MC dropout** uncertainty (mean *p*, variance across passes) | `src/modeling/infer_mc.py`; used when `--mc-samples > 0` |
| **Chip-level aggregation** (mean/median *p*, mean var, frac high-uncertainty) | `src/modeling/aggregate.py` → `aggregate.json` from infer CLI |
| **One-chip inference** CLI | `modeling/inference/infer_chip.py` |
| Local mirror for speed | `tools/sync_alphaearth_clips_from_s3.sh` + `ALPHA_EARTH_DATA_SOURCE=auto` |

### One-chip inference

```bash
export ALPHA_EARTH_DATA_SOURCE=s3   # or auto after sync
python modeling/inference/infer_chip.py \
  --checkpoint outputs/experiments/<run_id>/best.pt \
  --row-index 0 --split val \
  --mc-samples 20
```

Or `--local-path data/derived/alpha_earth_clips/.../chip.tif`. Outputs: `outputs/predictions/<run_id>/` — `mean_prob.npz`, optional `var_prob.npz`, `aggregate.json`.

## Next implementation steps (in order)

1. **GeoTIFF export** — Write `mean_prob` / `var` rasters with georeference from source chip (rasterio `profile`), not only NPZ.
2. **Batch infer** — Loop over `chips_index.csv` (or a manifest) and write one folder per chip or a single VRT/catalog for paper figures.
3. **Large-area tiled inference** — Sliding window over a big GeoTIFF, overlap + blend, stitch to one prob/uncertainty raster (Phase 5).
4. **Multi-farm / polygon zonal stats** — Rasterize or vector overlay: per-polygon mean *p*, uncertainty summary, tables for the “20 farmers” figure.
5. **Baselines & ablations** — Optional RF on embedding vectors (Earth Engine–style) for a comparison table; toggles for Dice-only / no-MC in the paper.
6. **Optional** — TensorBoard or Weights & Biases; config hash column in CSV; automatic `aws s3 sync` of `outputs/experiments/` after each run.

## Still optional / paper polish

- Calibration (temperature scaling) on val.
- Deep ensemble (multiple checkpoints) instead of or in addition to MC dropout.
- DeepLabv3+ or other backbone (currently U-Net only).
