#!/usr/bin/env bash
# Mirror AlphaEarth chips S3 → repo paths used in chips_index.csv (~1.5–2 GB for ~9.5k chips; fits Studio 50 GB).
# After sync: training uses local GeoTIFFs → fast DataLoader workers on train/val/test (no /vsis3/ in workers).
#
# Usage (from repo root):
#   ./tools/sync_alphaearth_clips_from_s3.sh
#   ./tools/sync_alphaearth_clips_from_s3.sh --dryrun
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
aws s3 sync "s3://${BUCKET}/${PREFIX}/" "${DEST}/" --only-show-errors "$@"
echo "Done."
echo "  export ALPHA_EARTH_DATA_SOURCE=auto   # prefer disk when file exists (recommended)"
echo "  export ALPHA_EARTH_DATA_SOURCE=local  # fail if any chip missing locally"
echo "Then train; val/test loaders auto-use parallel workers when chips are on disk."
