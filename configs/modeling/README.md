# Modeling configs

- **`tomato_unet.yaml`** — default training run: U-Net on AlphaEarth chip GeoTIFFs, pixelwise tomato vs non-tomato with masked BCE + Dice.

Override paths or hyperparameters by copying and editing, then:

```bash
python modeling/train/train.py --config configs/modeling/tomato_unet.yaml
```

From repo root with `PYTHONPATH` set to `.` (or `pip install -e .`). SageMaker / S3 / Studio: **`guide/README.md`**.
