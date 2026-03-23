#!/usr/bin/env python3
"""SageMaker training entry point stub. Replace with real training loop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/tmp/model")
    sm_channel_train = os.environ.get("SM_CHANNEL_TRAIN", "")
    sm_channel_val = os.environ.get("SM_CHANNEL_VALIDATION", "")

    print("Hyperparameters:", json.dumps(vars(args), indent=2))
    print("SM_MODEL_DIR:", sm_model_dir)
    print("SM_CHANNEL_TRAIN:", sm_channel_train)
    print("SM_CHANNEL_VALIDATION:", sm_channel_val)

    Path(sm_model_dir).mkdir(parents=True, exist_ok=True)
    (Path(sm_model_dir) / "stub_model.txt").write_text("Replace with checkpoint save.\n", encoding="utf-8")
    print("Stub training finished.")


if __name__ == "__main__":
    main()
