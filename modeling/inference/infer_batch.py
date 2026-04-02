#!/usr/bin/env python3
"""Batch inference over chips_index.csv (one subfolder per chip_id).

Example::

    export ALPHA_EARTH_DATA_SOURCE=auto
    python modeling/inference/infer_batch.py \\
      --checkpoint outputs/experiments/<run_id>/best.pt \\
      --split test --limit 50 \\
      --mc-samples 10 --geotiff
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from tqdm import tqdm

from src.modeling.dataset import load_chips_table
from src.modeling.infer_io import load_checkpoint
from src.modeling.infer_run import run_chip_forward, save_chip_outputs, tensors_from_row
from src.utils.paths import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch infer chips from chips_index.csv")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config-chips-csv", type=Path, default=None)
    ap.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=None, help="Max chips (default: all in split)")
    ap.add_argument("--mc-samples", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--geotiff", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="Start row index within split")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    data_cfg = cfg.get("data", {})
    target_hw = tuple(data_cfg.get("target_hw", [128, 128]))
    chips_csv = args.config_chips_csv or REPO_ROOT / str(data_cfg.get("chips_index_csv", "data/splits/chips_index.csv"))

    df = load_chips_table(chips_csv)
    sub = df[df["split"] == args.split].reset_index(drop=True)
    sub = sub.iloc[args.start :]
    if args.limit is not None:
        sub = sub.iloc[: args.limit]

    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "predictions" / f"batch_{args.checkpoint.parent.name}_{args.split}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []
    base_meta = {
        "checkpoint": str(args.checkpoint),
        "target_hw": list(target_hw),
        "mc_samples": int(args.mc_samples),
        "split": args.split,
    }

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"infer {args.split}"):
        try:
            lp, _su, tensors, chip_id = tensors_from_row(row, target_hw, REPO_ROOT)
            mean_p, var_p = run_chip_forward(model, device, tensors, args.mc_samples)
            save_chip_outputs(
                out_dir,
                chip_id,
                mean_p,
                var_p,
                tensors["mask"],
                base_meta,
                source_path=lp,
                write_geotiff=args.geotiff,
                flat_output=False,
            )
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in chip_id)[:200]
            agg_f = out_dir / safe / "aggregate.json"
            if agg_f.is_file():
                rows_out.append(json.loads(agg_f.read_text(encoding="utf-8")))
        except Exception as e:
            rows_out.append({"chip_id": str(row.get("chip_id", "")), "error": str(e)})

    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(rows_out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(rows_out)} entries under {out_dir}")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
