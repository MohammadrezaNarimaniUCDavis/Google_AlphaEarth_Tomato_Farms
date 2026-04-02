# AWS preflight (which account am I using?)

## What it does

`show-aws-context.ps1` sets **`AWS_PROFILE`** and **`AWS_DEFAULT_PROFILE`** for the current PowerShell session, then prints:

- `aws configure list`
- `aws sts get-caller-identity`
- `aws iam get-user` (only when the caller is an **IAM user**; skipped for roles)

Default profile for this repo: **`mohammadreza-digitalaglab`**.

## Override the default profile

1. **One-off:**  
   `.\show-aws-context.ps1 -Profile "other-profile"`

2. **Persistent (not committed):** create **`tools/aws-preflight/.default-profile`** (gitignored) containing a single line, e.g.  
   `mohammadreza-kobin`

## Run when Cursor opens this folder

The workspace includes a **task** that runs this script on **folder open** (trusted workspace). If it does not run automatically, run **Tasks: Run Task** → **AWS: show account context**.

To allow automatic tasks: **Settings** → search **Allow Automatic Tasks in Folder** → **On** (or use `.vscode/settings.json` in this repo).

## Manual

From repo root:

```powershell
.\tools\aws-preflight\show-aws-context.ps1
```

Environment variables apply only to that PowerShell process unless you dot-source or set them in your profile.

## Upload AlphaEarth GeoTIFF clips to S3

Mirrors **`data/derived/alpha_earth_clips/`** → **`s3://…/google-alphaearth-tomato-farms/derived/alpha_earth_clips/`** (see **`data/s3/README.md`**).

```powershell
.\tools\aws-preflight\sync-alphaearth-clips-to-s3.ps1
```

Dry-run: add **`-DryRun`**. Large first uploads can take hours; re-runs only send new/changed files.

## Cursor → SageMaker Studio (prerequisites only)

The Toolkit connection itself is **in Cursor** (GUI). From repo root, verify CLI / SSM plugin / SSH and list domains:

```powershell
.\tools\aws-preflight\check-cursor-sagemaker-prereqs.ps1
```

Step-by-step: **`guide/02-sagemaker-cursor-remote.md`**.

If Session Manager is installed but **not on PATH**, run once:

```powershell
.\tools\aws-preflight\add-session-manager-to-path-user.ps1
```

Restart Cursor afterward. Domain / user profile snapshot: **`guide/02-sagemaker-cursor-remote.md`** (table at top).
