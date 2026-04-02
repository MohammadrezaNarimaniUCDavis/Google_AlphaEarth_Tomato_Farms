#!/usr/bin/env python3
"""Train tomato vs non-tomato pixelwise model (AlphaEarth chips).

Run from repo root::

    python modeling/train/train.py --config configs/modeling/tomato_unet.yaml

On SageMaker, set ``SM_MODEL_DIR``; checkpoints and metrics are copied there.

Environment:

- ``ALPHA_EARTH_DATA_SOURCE``: ``auto`` (default), ``local``, or ``s3`` — how to open GeoTIFFs
  (see ``src/modeling/io_paths.py``).
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
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    out = train_model(cfg, repo_root=_REPO_ROOT)
    print("Experiment dir:", out)


if __name__ == "__main__":
    main()
