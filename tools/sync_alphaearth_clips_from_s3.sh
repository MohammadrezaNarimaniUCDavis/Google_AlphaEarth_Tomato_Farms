#!/usr/bin/env bash
# One-time (or occasional) mirror of AlphaEarth chips from project S3 → local paths in chips_index.csv.
# After sync: unset ALPHA_EARTH_DATA_SOURCE or use "auto" so training reads from disk (much faster than /vsis3/).
#
# Usage (from repo root):
#   chmod +x tools/sync_alphaearth_clips_from_s3.sh
#   ./tools/sync_alphaearth_clips_from_s3.sh
#
# Optional: DEST override (default matches CSV local_path prefix)
#   DEST=data/derived/alpha_earth_clips ./tools/sync_alphaearth_clips_from_s3.sh

set -euo pipefail
BUCKET="${S3_BUCKET:-tomato-alphaearth-054037103012-data}"
PREFIX="${S3_PREFIX:-google-alphaearth-tomato-farms/derived/alpha_earth_clips}"
DEST="${DEST:-data/derived/alpha_earth_clips}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p "${DEST}"
echo "Syncing s3://${BUCKET}/${PREFIX}/ → ${REPO_ROOT}/${DEST}/"
aws s3 sync "s3://${BUCKET}/${PREFIX}/" "${DEST}/" "$@"
echo "Done. For training use: export ALPHA_EARTH_DATA_SOURCE=auto"
echo "(or 'local' if you want to force disk-only)."
