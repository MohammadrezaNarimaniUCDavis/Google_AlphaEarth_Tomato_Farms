#!/usr/bin/env python3
"""Train tomato vs non-tomato pixelwise model (AlphaEarth chips).

Run from repo root::

    python modeling/train/train.py --config configs/modeling/tomato_unet.yaml

On SageMaker, set ``SM_MODEL_DIR``; checkpoints and key metrics are copied there.

Environment:

- ``ALPHA_EARTH_DATA_SOURCE``: ``auto`` (default), ``local``, or ``s3`` — how to open GeoTIFFs
  (see ``src/modeling/io_paths.py``).
- ``ALPHA_EARTH_EXPERIMENT_SYNC_S3``: optional ``s3://bucket/prefix/`` — after training, the
  experiment folder is synced to ``.../prefix/<run_id>/`` (requires ``aws s3 sync`` permissions).
- ``ALPHA_EARTH_TRAIN_NUM_WORKERS``: optional integer — overrides ``training.num_workers`` (use
  ``0``–``2`` if DataLoader workers exhaust RAM and the process dies mid-epoch with no traceback).
- ``ALPHA_EARTH_TRAIN_BATCH_SIZE``, ``ALPHA_EARTH_TRAIN_PREFETCH_FACTOR``: same pattern.
- ``ALPHA_EARTH_TRAIN_USE_AMP``: ``0``/``1`` — disable or force mixed precision (CUDA only).
- ``ALPHA_EARTH_TRAIN_EVAL_NUM_WORKERS``: overrides ``training.eval_num_workers`` for **val/test** loaders only
  (train-split metrics use the same workers as training for I/O speed).
- ``ALPHA_EARTH_S3_TRAIN_WORKERS_CAP``: max train ``num_workers`` when data is read from S3 (``0`` = main process only).
- ``ALPHA_EARTH_TRAIN_FAST=1``: shorthand preset (overridden by the more specific env vars above).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path (parent of ``modeling/``)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.modeling.train_config import load_yaml
from src.modeling.train_runner import train_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AlphaEarth tomato U-Net")
    ap.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "modeling" / "tomato_unet.yaml",
        help="Path to modeling YAML",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Override training.epochs in config")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size in config",
    )
    ap.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Cap training batches per epoch (smoke test; unset = full epoch)",
    )
    ap.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Cap batches for train-eval, val, and test passes (smoke test)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Shorthand: 1 epoch, batch 32, 8 train + 8 eval batches (CLI overrides win)",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to last.pt or best.pt under outputs/experiments/<run_id>/ to continue training",
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.smoke:
        tr = cfg.setdefault("training", {})
        tr["epochs"] = 1
        tr.setdefault("batch_size", 32)
        tr.setdefault("max_train_batches", 8)
        tr.setdefault("max_eval_batches", 8)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    if args.max_train_batches is not None:
        cfg.setdefault("training", {})["max_train_batches"] = int(args.max_train_batches)
    if args.max_eval_batches is not None:
        cfg.setdefault("training", {})["max_eval_batches"] = int(args.max_eval_batches)
    out = train_model(
        cfg,
        repo_root=_REPO_ROOT,
        resume_from=args.resume,
        command_argv=sys.argv,
    )
    print("Experiment dir:", out)


if __name__ == "__main__":
    main()
