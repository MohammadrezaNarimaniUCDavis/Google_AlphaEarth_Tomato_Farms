#!/usr/bin/env bash
# SageMaker Studio: sync chips (~1.5 GB) then print a fast training command (resume-friendly).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== 1) Mirror chips from S3 (needs aws cli + s3:GetObject) ==="
"${SCRIPT_DIR}/sync_alphaearth_clips_from_s3.sh"

echo ""
echo "=== 2) Quick check (first train chip from disk) ==="
export ALPHA_EARTH_DATA_SOURCE=auto
python3 tools/check_training_ready.py

echo ""
echo "=== 3) Start training (edit RESUME path if needed) ==="
cat << 'EOS'
export ALPHA_EARTH_DATA_SOURCE=auto
export ALPHA_EARTH_EXPERIMENT_SYNC_S3=s3://tomato-alphaearth-054037103012-data/google-alphaearth-tomato-farms/experiments/
unset ALPHA_EARTH_TRAIN_FAST
export ALPHA_EARTH_TRAIN_BATCH_SIZE=40
export ALPHA_EARTH_TRAIN_NUM_WORKERS=8
export ALPHA_EARTH_TRAIN_PREFETCH_FACTOR=8
# With full local mirror, S3 cap is not applied; val/test use 8 workers automatically.

nohup python3 -u modeling/train/train.py \
  --config configs/modeling/tomato_unet_studio_fast.yaml \
  --resume outputs/experiments/20260402T230522Z/last.pt \
  >> outputs/train_full.log 2>&1 &

./tools/tail_train_log.sh
EOS
