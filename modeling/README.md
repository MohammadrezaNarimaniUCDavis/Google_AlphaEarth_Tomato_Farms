# Modeling

## Layout

- `train/` — Scripts invoked by SageMaker (`train.py` as entry point). Reads hyperparameters from CLI / environment variables SageMaker provides.
- `inference/` — Batch transform or real-time endpoint code (add when you have a trained artifact).

## Workflow

1. Upload clipped tensors or manifests under `data/derived/` to S3 (or mount EFS in Studio).
2. Point SageMaker `channels` at those prefixes (see `configs/sagemaker.example.yaml`).
3. Run a SageMaker `Estimator` with `source_dir` containing `modeling/train` and `entry_point=train.py`.
4. Save evaluation figures to `/opt/ml/output/data/` in the training job (SageMaker) and copy to repo `figures/` for the paper.

The stub `train/train.py` only prints received paths and hyperparameters so you can verify wiring before adding PyTorch/TensorFlow.
