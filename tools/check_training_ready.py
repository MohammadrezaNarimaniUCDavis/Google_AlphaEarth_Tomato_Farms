#!/usr/bin/env python3
"""Verify Studio/SageMaker is ready to run modeling/train/train.py.

Checks: PyTorch + CUDA, rasterio, optional local chip, or S3 chip read
(``ALPHA_EARTH_DATA_SOURCE=s3``).

Exit code 0 only if at least one chip can be opened. Otherwise prints IAM fix path.

Usage (repo root)::

    python tools/check_training_ready.py
    ALPHA_EARTH_DATA_SOURCE=s3 python tools/check_training_ready.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    print("=== AlphaEarth training preflight ===")
    try:
        import torch

        print(f"PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print("FAIL: PyTorch not installed.", e)
        return 1

    try:
        import rasterio  # noqa: F401
    except ImportError as e:
        print("FAIL: rasterio not installed.", e)
        return 1

    os.chdir(_REPO)
    from src.modeling.dataset import AlphaEarthChipSegDataset, load_chips_table
    from src.utils.paths import REPO_ROOT

    csv_path = REPO_ROOT / "data" / "splits" / "chips_index.csv"
    if not csv_path.is_file():
        print(f"FAIL: Missing {csv_path}")
        return 1

    df = load_chips_table(csv_path)
    mode = os.environ.get("ALPHA_EARTH_DATA_SOURCE", "auto").strip().lower()
    print(f"ALPHA_EARTH_DATA_SOURCE={mode!r}  (use 's3' on Studio if chips are only in S3)")

    ds = AlphaEarthChipSegDataset(df, "train", (128, 128), augment=False)
    try:
        _ = ds[0]
    except Exception as e:
        print("FAIL: Cannot read first training chip:", e)
        print()
        print("Fix: attach S3 read policy to your SageMaker execution role, e.g.")
        print("  tools/aws-preflight/sagemaker-execution-s3-tomato-bucket-policy.json")
        print("See: guide/02-sagemaker-cursor-remote.md (IAM fix: s3:GetObject)")
        return 1

    print("OK: Opened first train chip — you can run:")
    print("  export ALPHA_EARTH_DATA_SOURCE=s3   # if using S3")
    print("  python modeling/train/train.py --config configs/modeling/tomato_unet.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
