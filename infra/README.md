# SageMaker notebook (Infrastructure as Code)

AWS cannot be configured from GitHub or from this repo automatically. Use this template **from your PC** with the AWS CLI and your credentials.

## What it creates

- **S3 bucket** `tomato-alphaearth-<ACCOUNT_ID>-data` (retained if you delete the stack). Use prefix **`google-alphaearth-tomato-farms/`** inside it for this repo — see **`data/s3/README.md`** and **`configs/paths.example.yaml`** → `s3:`.
- **IAM role** with **AmazonSageMakerFullAccess** (same broad access as the console “easy” path)
- **Notebook instance** `digitalaglab-tomato-alphaearth`, **ml.g4dn.xlarge**, **50 GB**, **JupyterLab 4** (`notebook-al2-v3`), **IMDSv2**

## Prerequisites

- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed
- Credentials for the target account (e.g. profile `mohammadreza-digitalaglab`)
- Permission to create IAM roles, S3 buckets, and SageMaker notebook instances
- **Service quota** for `ml.g4dn.xlarge` in your region (request increase if create fails)

## Deploy (Windows)

From the repository root:

```powershell
.\infra\deploy-sagemaker-notebook.ps1
```

Override profile or region if needed:

```powershell
.\infra\deploy-sagemaker-notebook.ps1 -Profile "mohammadreza-digitalaglab" -Region "us-west-2"
```

Equivalent without the script:

```text
aws cloudformation deploy --template-file infra/sagemaker-notebook.yaml --stack-name tomato-alphaearth-sagemaker-notebook --capabilities CAPABILITY_IAM --region us-west-2
```

## After deploy

1. **Console** → **SageMaker** → **Notebook instances** → wait for **InService** → **Open JupyterLab**.
2. Clone your private repo in the terminal (HTTPS + PAT or SSH).
3. Use the **Outputs** bucket name for large files; keep the notebook disk for code and cache.
4. **Stop** the notebook when not in use.

## Cursor, SSH, and local folder layout

See [`tools/sagemaker-remote/README.md`](../tools/sagemaker-remote/README.md): **Notebook Instance ≠ Studio remote IDE**; use Git sync + browser Jupyter for the current stack, or move to **Studio spaces** for AWS’s Cursor + SSH flow.

## Troubleshooting

- **Notebook name already exists**: change `NotebookInstanceName` in `sagemaker-notebook.yaml` or delete the old instance in the console, then redeploy.
- **Insufficient capacity / quota**: try another AZ/region or request a **quota increase** for G4 instances.
- **Tighter IAM later**: replace `AmazonSageMakerFullAccess` with scoped policies and specific S3 ARNs.
