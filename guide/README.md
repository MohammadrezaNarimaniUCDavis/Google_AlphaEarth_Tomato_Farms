# Guide — AlphaEarth tomato modeling (read this on **remote** or **local** Cursor)

Use this folder when you open the repo in **SageMaker Studio via Cursor** (remote window). Local laptop work: prep, LandIQ notebooks, optional CPU smoke tests. **GPU training** runs on Studio.

**Open this repo as the Cursor workspace:** **File → Open Folder…** → select the **`Google_AlphaEarth_Tomato_Farms`** directory (repo root), **or** **File → Open Workspace from File…** → choose **`Google_AlphaEarth_Tomato_Farms.code-workspace`** in that same directory. If Cursor opened your home folder (`~`) instead, switch to the repo root so paths, git, and the terminal default match the project.

| Doc | Purpose |
|-----|---------|
| **[01-project-plan.md](01-project-plan.md)** | Scientific goal, modeling story, metrics, phases (paper framing). |
| **[02-sagemaker-cursor-remote.md](02-sagemaker-cursor-remote.md)** | Domain, Code Editor space, **IAM `StartSession` fix**, Open in Cursor, costs, preflight scripts. |
| **[03-data-s3-and-training.md](03-data-s3-and-training.md)** | S3 layout, `chips_index`, env vars, install PyTorch on Studio, `train.py`, outputs. |

**Quick start (Studio terminal, after `git pull`):**

```bash
cd Google_AlphaEarth_Tomato_Farms   # or your clone path
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-modeling-cpu.txt
pip install -e . --no-deps
export ALPHA_EARTH_DATA_SOURCE=s3
python modeling/train/train.py --config configs/modeling/tomato_unet.yaml
```

**Note:** **Open in Cursor** opens a **new** Cursor window attached to Studio; that is normal. Use that window for training; local window is optional.

**If `git push` / `git pull` from Studio fails (no GitHub credentials):** use a **git bundle** from another machine, or copy the bundle from Studio’s persistent file system. Example rehydrate: `git clone /path/to/Google_AlphaEarth_Tomato_Farms-main-*.bundle ./Google_AlphaEarth_Tomato_Farms` then `cd Google_AlphaEarth_Tomato_Farms && git checkout main && git remote add origin https://github.com/.../Google_AlphaEarth_Tomato_Farms.git` for future pushes.

Code layout: **`modeling/README.md`**, **`src/modeling/`**, **`configs/modeling/`**.
