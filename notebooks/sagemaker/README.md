# SageMaker notebooks and jobs

## Studio / Notebook instance

1. Clone this repository (or sync `modeling/`, `src/`, and `configs/`).
2. Use a GPU instance (for example `ml.g4dn.xlarge`) when training deep models.
3. Install dependencies: `pip install -r requirements.txt` (add `torch` or `tensorflow` as needed).

## Data channels

- Upload `data/derived/` subsets to S3.
- Set `SM_CHANNEL_TRAIN` and `SM_CHANNEL_VALIDATION` via the SageMaker estimator, or read paths from `hyperparameters`.

## IAM

Create an execution role that allows SageMaker to read your S3 buckets and write model artifacts. Put the role ARN in a **local** copy of `configs/sagemaker.example.yaml` (do not commit secrets or private bucket names if the repo is public).

## Figures

In training scripts, write diagnostics under `/opt/ml/output/data/` so they appear in the job output in S3; download into repo `figures/` for LaTeX.

## Entry point

Start from `modeling/train/train.py` and replace the stub with your training code.
