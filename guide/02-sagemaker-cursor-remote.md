# SageMaker Studio + Cursor (remote IDE)

## References

- [Cursor / Kiro remote to Studio — AWS What’s New (Mar 26, 2026)](https://aws.amazon.com/about-aws/whats-new/2026/03/amazon-sagemaker-studio-kiro-cursor/)
- [Connect local VS Code to Studio spaces](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-vscode.html) (same class of setup: Toolkit + Remote SSH + Session Manager)

## This project’s Studio resources (us-west-2)

| Item | Value |
|------|--------|
| Region | `us-west-2` |
| Domain ID | `d-gfmncrlaaors` |
| Domain name | `QuickSetupDomain-20260402T114484` |
| Studio URL | `https://d-gfmncrlaaors.studio.us-west-2.sagemaker.aws` |
| User profile | `default-20260402T114484` |
| Code Editor space (example) | `tomato-ml` |

## Code Editor space settings (tomato ML)

| Setting | Value |
|---------|--------|
| **Instance** | `ml.g4dn.xlarge` (GPU; avoid `ml.t3.medium` for remote IDE — under 8 GiB RAM not supported for remote) |
| **Image** | SageMaker Distribution 3.9.x (GPU image as offered) |
| **Storage** | 15 GB (or 10–20 GB) |
| **Remote access** | **On** (required for **Open in Cursor**) |

**Run space** → **Running** → **Open in Cursor**. A **new** Cursor window opens attached to Studio; use that window for the repo and terminal.

## IAM fix: `Remote access denied` / `sagemaker:StartSession`

If you see:

`User: ... assumed-role/AmazonSageMaker-ExecutionRole-.../SageMaker is not authorized to perform: sagemaker:StartSession on resource: ... space/...`

Attach an **inline policy** to role **`AmazonSageMaker-ExecutionRole-20260402T114485`** (match your account’s execution role name in the error):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "StudioSpaceRemoteStartSession",
      "Effect": "Allow",
      "Action": "sagemaker:StartSession",
      "Resource": "arn:aws:sagemaker:us-west-2:054037103012:space/d-gfmncrlaaors/*"
    }
  ]
}
```

Replace account ID if yours differs. Retry **Open in Cursor** after IAM propagation.

## IAM fix: training cannot read chips from S3 (`AccessDenied` on `s3:GetObject`)

If `rasterio` / training fails with **User: ... AmazonSageMaker-ExecutionRole-... is not authorized to perform: s3:GetObject** on `tomato-alphaearth-054037103012-data/...`, attach an **inline policy** to the same **SageMaker execution role** (the one shown in the error).

Use the JSON in the repo (adjust bucket name or account if you changed them):

**[`tools/aws-preflight/sagemaker-execution-s3-tomato-bucket-policy.json`](../tools/aws-preflight/sagemaker-execution-s3-tomato-bucket-policy.json)**

It allows **read** on `google-alphaearth-tomato-farms/*` and **write** on `google-alphaearth-tomato-farms/experiments/*` (optional: for syncing checkpoints/metrics to S3). Wait a minute after saving the policy, then retry training with `ALPHA_EARTH_DATA_SOURCE=s3`.

## Local laptop prerequisites (before Open in Cursor)

1. **Cursor** + extensions: **AWS Toolkit**, **Remote - SSH**.
2. **AWS CLI v2** — profile (e.g. `mohammadreza-digitalaglab`) in `us-west-2`.
3. **Session Manager plugin** — [install guide](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html). On Windows, if not on PATH:  
   `.\tools\aws-preflight\add-session-manager-to-path-user.ps1`  
   then restart Cursor.
4. **OpenSSH** client.

Verify:

```powershell
.\tools\aws-preflight\check-cursor-sagemaker-prereqs.ps1
```

## Toolkit sign-in

- **IAM Credential** (not Identity Center) if you use access-key profile.
- Profile + region **`us-west-2`**.

## Cost

- **GPU** (`g4dn`) bills while the **space app is running**. **Stop** the Code Editor app when idle.
- **Space** may remain with **storage** only — much cheaper than GPU hours.

## Check if GPU is still running (CLI)

```bash
aws sagemaker list-apps --domain-id-equals d-gfmncrlaaors --region us-west-2 --profile YOUR_PROFILE
```

App status **`Deleted`** means compute is stopped; space can still be **`InService`** (storage only).
