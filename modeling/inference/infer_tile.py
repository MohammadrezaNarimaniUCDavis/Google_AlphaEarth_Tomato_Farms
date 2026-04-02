#!/usr/bin/env python3
"""Sliding-window inference on a large multi-band GeoTIFF (Phase 5).

Patches are resized to the model ``target_hw`` (same as training), predictions blended
with overlap-weighted average back to full raster grid.

Example::

    export ALPHA_EARTH_DATA_SOURCE=auto
    python modeling/inference/infer_tile.py \\
      --checkpoint outputs/experiments/<run_id>/best.pt \\
      --input path/to/large_alphaearth.tif \\
      --overlap 32 \\
      --out outputs/predictions/region_prob.tif \\
      --mc-samples 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from src.modeling.infer_io import load_checkpoint
from src.modeling.tile_infer import infer_large_geotiff, write_raster_from_meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiled inference on large GeoTIFF")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="Large multi-band GeoTIFF (same band count as training)")
    ap.add_argument("--out", type=Path, required=True, help="Output mean probability GeoTIFF")
    ap.add_argument("--out-var", type=Path, default=None, help="Optional output variance GeoTIFF (if --mc-samples > 0)")
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--tile-h", type=int, default=None, help="Window height (default: model target_h)")
    ap.add_argument("--tile-w", type=int, default=None, help="Window width (default: model target_w)")
    ap.add_argument("--mc-samples", type=int, default=0)
    args = ap.parse_args()

    inp = args.input
    if not inp.is_absolute():
        inp = _REPO / inp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    target_hw = tuple(cfg.get("data", {}).get("target_hw", [128, 128]))

    mean, var, meta = infer_large_geotiff(
        model,
        device,
        inp,
        target_hw,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        overlap=args.overlap,
        mc_samples=args.mc_samples,
    )
    out = args.out
    if not out.is_absolute():
        out = _REPO / out
    write_raster_from_meta(mean, meta, out)
    print("Wrote", out)
    if var is not None and args.out_var:
        ov = args.out_var
        if not ov.is_absolute():
            ov = _REPO / ov
        write_raster_from_meta(var, meta, ov)
        print("Wrote", ov)


if __name__ == "__main__":
    main()
