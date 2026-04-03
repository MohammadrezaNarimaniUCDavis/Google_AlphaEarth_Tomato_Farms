#!/usr/bin/env bash
# Follow training log with tqdm lines broken onto separate rows.
# Usage: ./tools/tail_train_log.sh [path/to/train.log]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${1:-$ROOT/outputs/train_full.log}"
exec tail -f "$LOG" | tr '\r' '\n'
