#!/usr/bin/env python3
"""Run trained U-Net on one AlphaEarth chip: probability map + optional MC-dropout uncertainty.

Examples::

    python modeling/inference/infer_chip.py \\
      --checkpoint outputs/experiments/<run_id>/best.pt \\
      --row-index 0 --split val --geotiff

    python modeling/inference/infer_chip.py \\
      --checkpoint outputs/experiments/<run_id>/best.pt \\
      --local-path data/derived/alpha_earth_clips/ee/.../chip.tif

With local mirror (fast): ``export ALPHA_EARTH_DATA_SOURCE=auto``

Outputs under ``--out-dir``: ``mean_prob.npz``, optional ``var_prob.npz``, ``aggregate.json``,
and with ``--geotiff``: ``pred_mean_prob.tif`` / ``pred_var_prob.tif`` (same bounds as source chip).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd
import torch

from src.modeling.dataset import load_chip_for_model, load_chips_table
from src.modeling.infer_io import load_checkpoint
from src.modeling.infer_run import run_chip_forward, save_chip_outputs
from src.utils.paths import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer tomato probability (+ optional MC uncertainty) on one chip")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or last.pt")
    ap.add_argument("--config-chips-csv", type=Path, default=None, help="Override chips index CSV from checkpoint cfg")
    ap.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    ap.add_argument("--row-index", type=int, default=None, help="Row index within split (uses chips CSV)")
    ap.add_argument("--local-path", type=Path, default=None, help="GeoTIFF path (repo-relative or absolute)")
    ap.add_argument("--s3-uri", type=str, default=None, help="Optional s3:// URI if file not local")
    ap.add_argument("--mc-samples", type=int, default=0, help="MC dropout passes (0 = single eval forward)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    ap.add_argument("--geotiff", action="store_true", help="Write pred_mean_prob.tif (+ var if MC) georeferenced like source")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    data_cfg = cfg.get("data", {})
    target_hw = tuple(data_cfg.get("target_hw", [128, 128]))
    chips_csv = args.config_chips_csv or REPO_ROOT / str(data_cfg.get("chips_index_csv", "data/splits/chips_index.csv"))

    if args.local_path is not None:
        lp = args.local_path
        if not lp.is_absolute():
            lp = REPO_ROOT / lp
        tensors = load_chip_for_model(lp, target_hw, s3_uri=args.s3_uri)
        chip_id = lp.stem
    elif args.row_index is not None:
        df = load_chips_table(chips_csv)
        sub = df[df["split"] == args.split].reset_index(drop=True)
        if args.row_index < 0 or args.row_index >= len(sub):
            raise SystemExit(f"row-index {args.row_index} out of range for split {args.split} (n={len(sub)})")
        row = sub.iloc[args.row_index]
        lp = Path(row["local_path"])
        if not lp.is_absolute():
            lp = REPO_ROOT / lp
        su = None
        if "s3_uri" in row.index and pd.notna(row.get("s3_uri")):
            s = str(row["s3_uri"]).strip()
            su = s if s.startswith("s3://") else None
        tensors = load_chip_for_model(lp, target_hw, s3_uri=su)
        chip_id = str(row.get("chip_id", args.row_index))
    else:
        raise SystemExit("Provide --local-path or --row-index")

    mean_p, var_p = run_chip_forward(model, device, tensors, args.mc_samples)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = REPO_ROOT / "outputs" / "predictions" / args.checkpoint.parent.name
    out_dir = Path(out_dir)

    base_meta = {
        "checkpoint": str(args.checkpoint),
        "target_hw": list(target_hw),
        "mc_samples": int(args.mc_samples),
    }
    save_chip_outputs(
        out_dir,
        chip_id,
        mean_p,
        var_p,
        tensors["mask"],
        base_meta,
        source_path=lp,
        write_geotiff=args.geotiff,
        flat_output=True,
    )
    agg_path = out_dir / "aggregate.json"
    if agg_path.is_file():
        print(agg_path.read_text(encoding="utf-8"))
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
