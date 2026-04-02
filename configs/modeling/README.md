# Modeling configs

- **`tomato_unet.yaml`** — default training run: U-Net on AlphaEarth chip GeoTIFFs, pixelwise tomato vs non-tomato with masked BCE + Dice.
- **`tomato_unet_smoke.yaml`** — same model, but caps batches per phase for a quick pipeline check.

`training.num_workers` (default **4**) and `prefetch_factor` overlap **S3/disk** reads with the GPU; set `num_workers: 0` only if you hit multiprocessing errors with rasterio.

Override paths or hyperparameters by copying and editing, then:

```bash
python modeling/train/train.py --config configs/modeling/tomato_unet.yaml
```

From repo root with `PYTHONPATH` set to `.` (or `pip install -e .`). SageMaker / S3 / Studio: **`guide/README.md`**.
