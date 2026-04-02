# Operations handbook — costs, GitHub, training, inference

Use this document to **resume work later** on SageMaker Studio + Cursor, control spend, and sync code with GitHub.

---

## 1. Cost (approximate, verify in your account)

### GPU Code Editor space (`ml.g4dn.xlarge`)

AWS publishes **on-demand** pricing for SageMaker **Code Editor** / JupyterLab **spaces** using **`ml.g4dn.xlarge`**. In AWS’s own pricing examples, the rate shown is **about USD $0.74 per hour** for that instance type (list price; **your invoice can differ** by region, purchase model, and discounts).

- **You pay for GPU compute while the space app is *running*** (kernel up), not only while `train.py` is active. **Closing Cursor does not stop the bill** if the Code Editor app is still **Running** in Studio.
- **Stop** the Code Editor application when idle to avoid burning GPU hours.
- **Space storage** (e.g. 15 GB) can incur **small ongoing storage charges** even when the app is stopped; deleting the space removes that footprint (only if you no longer need its disk).

**Rough examples (at ~$0.74/hr, illustrative only):**

| Hours running | Approx. compute (USD) |
|---------------|------------------------|
| 1 | ~$0.74 |
| 8 (one working day) | ~$5.9 |
| 24 | ~$18 |
| 120 (e.g. 5 days × 24 h) | ~$89 |

**Always confirm** in:

- [AWS Billing → Cost Explorer](https://console.aws.amazon.com/cost-management/home)
- [AWS Pricing Calculator](https://calculator.aws/)
- Official page: [Amazon SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/) (search for **ml.g4dn.xlarge** and **Code Editor** / **Studio** context)

### Other costs (usually smaller than GPU if training is moderate)

| Item | Notes |
|------|--------|
| **S3** | Storage (GB-month) + **GET/PUT** requests; many training reads can add a little; **syncing chips to disk** reduces repeated S3 GET during training. |
| **Data transfer** | Same **Region** (e.g. Studio and bucket in **us-west-2**) is typically low; cross-region or internet egress can add up. |
| **Domain / idle** | No substitute for checking **Cost Explorer** filters by service **SageMaker** and **S3**. |

### Cost control checklist

1. **Stop** Code Editor GPU app when not training or coding.  
2. Prefer **local/EFS mirror** of chips + `ALPHA_EARTH_DATA_SOURCE=auto` → **shorter wall time** → **fewer GPU hours** for the same experiment.  
3. Use **`tomato_unet_smoke.yaml`** before long runs to validate the pipeline.  
4. **Archive** old `outputs/experiments/` to S3 and delete huge local runs if disk is tight (optional).

---

## 2. GitHub — push and pull

### On your laptop (recommended when Studio has no token)

```bash
cd /path/to/Google_AlphaEarth_Tomato_Farms
git checkout main
git pull origin main
# … make changes …
git add -A && git commit -m "your message"
git push origin main
```

### On SageMaker Studio (if HTTPS fails with “could not read Username”)

Options:

1. **SSH remote** + deploy key or personal SSH key.  
2. **HTTPS + Personal Access Token** (PAT) with `git credential` or `GIT_ASKPASS`.  
3. **Push from laptop only**; on Studio use `git pull` if credentials work.

### After cloning on a new machine

```bash
git clone https://github.com/MohammadrezaNarimaniUCDavis/Google_AlphaEarth_Tomato_Farms.git
cd Google_AlphaEarth_Tomato_Farms
git checkout main
git pull
```

---

## 3. Environment setup (Studio, GPU)

```bash
cd Google_AlphaEarth_Tomato_Farms
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121  # or use image default
pip install -r requirements-modeling-cpu.txt   # includes fiona, rasterio, etc.
pip install -e . --no-deps
```

**IAM:** execution role needs **S3 read** (and optional **write** on `experiments/`) — see `tools/aws-preflight/sagemaker-execution-s3-tomato-bucket-policy.json` and **`guide/02-sagemaker-cursor-remote.md`**.

**Preflight:**

```bash
export ALPHA_EARTH_DATA_SOURCE=s3   # or auto after sync
python tools/check_training_ready.py
```

---

## 4. Data: S3 vs local mirror (speed and cost)

**One-time sync** (from repo root):

```bash
./tools/sync_alphaearth_clips_from_s3.sh
export ALPHA_EARTH_DATA_SOURCE=auto
```

Training:

```bash
python modeling/train/train.py --config configs/modeling/tomato_unet.yaml
```

**Smoke test** (few batches):

```bash
python modeling/train/train.py --config configs/modeling/tomato_unet_smoke.yaml
```

**Outputs:** `outputs/experiments/<run_id>/` — `metrics_epoch.csv`, `best.pt`, confusion JSONs, `metrics_test.json` (see **`guide/03-data-s3-and-training.md`**).

**Long run in background:**

```bash
mkdir -p outputs
export ALPHA_EARTH_DATA_SOURCE=auto
nohup python3 -u modeling/train/train.py --config configs/modeling/tomato_unet.yaml >> outputs/train_full.log 2>&1 &
tail -f outputs/train_full.log
```

**Stop training:**

```bash
pkill -f "modeling/train/train.py"
```

---

## 5. Inference pipeline (after you have `best.pt`)

| Step | Command / doc |
|------|----------------|
| One chip | `modeling/inference/infer_chip.py` |
| Many chips | `modeling/inference/infer_batch.py` |
| Large GeoTIFF | `modeling/inference/infer_tile.py` |
| Zonal stats (farms) | `modeling/inference/zonal_stats.py` |

Full detail: **`guide/04-inference-and-roadmap.md`**.

---

## 6. Documentation map (start here)

| Doc | Purpose |
|-----|---------|
| **`guide/README.md`** | Index + quick start |
| **`guide/01-project-plan.md`** | Science / paper framing |
| **`guide/02-sagemaker-cursor-remote.md`** | Studio, Cursor, IAM (`StartSession`, S3 policy) |
| **`guide/03-data-s3-and-training.md`** | Bucket layout, sync, training, metric caveats |
| **`guide/04-inference-and-roadmap.md`** | Inference CLIs, tiling, zonal stats, S3 archive |
| **`guide/05-operations-costs-and-handbook.md`** | **This file** — costs, GitHub, day-to-day ops |
| **`modeling/README.md`** | Train / infer entrypoints |

---

## 7. Repo layout (modeling)

```
configs/modeling/tomato_unet.yaml       # full training
configs/modeling/tomato_unet_smoke.yaml # fast sanity check
modeling/train/train.py                 # training CLI
modeling/inference/*.py                 # infer_chip, infer_batch, infer_tile, zonal_stats
src/modeling/                           # dataset, model, metrics, tile_infer, etc.
tools/sync_alphaearth_clips_from_s3.sh
tools/check_training_ready.py
```

---

## 8. Metrics and publishing

- Do **not** rely on **smoke** or **epoch-1** numbers alone for papers — use **full** training and **held-out test**.  
- See **`guide/03-data-s3-and-training.md`** (metrics caveats).

---

*Last updated to match repo workflow; pricing figures are indicative — confirm in AWS Billing and the official SageMaker pricing page.*
