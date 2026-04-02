# Inference

## Chip-level (implemented)

```bash
python modeling/inference/infer_chip.py \
  --checkpoint outputs/experiments/<run_id>/best.pt \
  --row-index 0 --split val \
  --mc-samples 20
```

See **`guide/04-inference-and-roadmap.md`** for outputs and next steps (GeoTIFF export, tiling, SageMaker batch transform).
