"""Dataset: AlphaEarth multi-band GeoTIFF chips with chip-level tomato / non-tomato labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from src.modeling.io_paths import resolve_raster_path


def _resize_stack(
    x: np.ndarray,
    mask_hw: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear resize for x (C,H,W), nearest for mask (H,W)."""
    import torch.nn.functional as F

    t = torch.from_numpy(x).float().unsqueeze(0)  # 1,C,h,w
    m = torch.from_numpy(mask_hw).float().unsqueeze(0).unsqueeze(0)  # 1,1,h,w
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    m = F.interpolate(m, size=(target_h, target_w), mode="nearest")
    return t.squeeze(0).numpy(), m.squeeze(0).squeeze(0).numpy()


class AlphaEarthChipSegDataset(Dataset):
    """One chip → resized tensor and pixel labels (uniform per chip) with valid mask."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        target_hw: tuple[int, int],
        *,
        augment: bool = False,
    ) -> None:
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.h, self.w = target_hw
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        local_path = row["local_path"]
        if "s3_uri" in row.index and pd.notna(row["s3_uri"]):
            s3_uri = str(row["s3_uri"]).strip()
            if not s3_uri.startswith("s3://"):
                s3_uri = None
        else:
            s3_uri = None
        rpath = resolve_raster_path(local_path, str(s3_uri) if s3_uri else None)

        with rasterio.open(rpath) as ds:
            arr = ds.read(out_dtype="float32")  # C,H,W
            if arr.ndim != 3:
                raise ValueError(f"Expected CHW array, got shape {arr.shape} for {rpath}")

        valid = np.isfinite(arr).all(axis=0).astype(np.float32)  # H,W
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        label = 1.0 if str(row["class_label"]).lower() == "tomato" else 0.0

        arr, valid = _resize_stack(arr, valid, self.h, self.w)
        y = np.full((1, self.h, self.w), label, dtype=np.float32) * valid[np.newaxis, ...]

        x = torch.from_numpy(arr).float()
        y_t = torch.from_numpy(y).float()
        m_t = torch.from_numpy(valid).float().unsqueeze(0)

        if self.augment and np.random.rand() < 0.5:
            x = torch.flip(x, dims=[2])
            y_t = torch.flip(y_t, dims=[2])
            m_t = torch.flip(m_t, dims=[2])
        if self.augment and np.random.rand() < 0.5:
            x = torch.flip(x, dims=[1])
            y_t = torch.flip(y_t, dims=[1])
            m_t = torch.flip(m_t, dims=[1])

        return {
            "x": x,
            "y": y_t,
            "mask": m_t,
            "chip_id": str(row.get("chip_id", idx)),
        }


def load_chips_table(chips_csv: Path) -> pd.DataFrame:
    if not chips_csv.is_file():
        raise FileNotFoundError(
            f"Missing {chips_csv}. Run: python tools/build_chips_index.py"
        )
    df = pd.read_csv(chips_csv)
    required = {"local_path", "class_label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"chips_index missing columns: {missing}")
    return df


def infer_in_channels(sample_path: Path | str, s3_uri: str | None = None) -> int:
    rpath = resolve_raster_path(Path(sample_path), s3_uri)
    with rasterio.open(rpath) as ds:
        return int(ds.count)
