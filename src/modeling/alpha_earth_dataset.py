"""Dataset and utilities for AlphaEarth tomato vs non_tomato chips.

Uses the chips index built by ``tools/build_chips_index.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.paths import REPO_ROOT, load_paths_config, resolve_under_root


@dataclass
class ChipSample:
    """Container for one chip sample."""

    image: Tensor  # (C, H, W), float32
    valid_mask: Tensor  # (1, H, W), bool
    label: Tensor  # scalar, 0 or 1 (non_tomato / tomato)
    meta: Dict[str, Any]


class AlphaEarthChipsDataset(Dataset):
    """PyTorch Dataset over the chips index (tomato vs non_tomato).

    It loads from **local GeoTIFF paths** pointed to by ``chips_index``.
    Use the S3 URIs in the same index when running on SageMaker if you
    want to stream from S3 instead of local disk.
    """

    def __init__(
        self,
        index_path: Path,
        split: str,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = load_paths_config()
        self.cfg = cfg
        self.root = REPO_ROOT

        if not index_path.is_file():
            raise FileNotFoundError(f"chips_index not found: {index_path}")

        if index_path.suffix == ".parquet":
            df = pd.read_parquet(index_path)
        else:
            df = pd.read_csv(index_path)

        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        df = df[df["split"].str.lower() == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows for split={split!r} in {index_path}")

        self.df = df

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.df)

    def _label_to_tensor(self, label: str) -> Tensor:
        # Binary: tomato = 1, non_tomato = 0
        y = 1 if label.lower() == "tomato" else 0
        return torch.tensor(y, dtype=torch.long)

    def _load_image_and_mask(self, path: Path) -> Tuple[Tensor, Tensor]:
        """Load GeoTIFF as (C,H,W) tensor and valid-pixel mask (1,H,W).

        Pixels with any NaN across bands are treated as invalid.
        """
        with rasterio.open(path) as ds:
            arr = ds.read()  # shape: (C, H, W)
            # Rasterio returns NaNs preserved; build mask where all bands are finite.
            valid = np.all(np.isfinite(arr), axis=0)  # (H, W)
            # Replace invalid with 0 so they don't explode numerically.
            arr[:, ~valid] = 0.0

        img = torch.from_numpy(arr.astype("float32"))
        mask = torch.from_numpy(valid.astype("bool")).unsqueeze(0)  # (1,H,W)
        return img, mask

    def __getitem__(self, idx: int) -> ChipSample:  # type: ignore[override]
        row = self.df.iloc[idx]
        rel = Path(str(row["local_path"]))
        full_path = resolve_under_root(str(rel), self.root)
        if not full_path.is_file():
            raise FileNotFoundError(f"GeoTIFF not found: {full_path} (from {rel})")

        image, valid_mask = self._load_image_and_mask(full_path)
        label = self._label_to_tensor(str(row["class_label"]))

        meta = {
            "chip_id": row.get("chip_id"),
            "class_label": row.get("class_label"),
            "split": row.get("split"),
            "local_path": str(full_path),
            "s3_uri": row.get("s3_uri"),
        }

        return ChipSample(image=image, valid_mask=valid_mask, label=label, meta=meta)


def collate_chips(batch: List[ChipSample]) -> ChipSample:
    """Collate function to batch ChipSample objects for a DataLoader.

    Returns a ChipSample where ``image``, ``valid_mask`` and ``label`` are
    batched tensors and ``meta`` is a list of per-item metadata dicts.
    """
    images = torch.stack([b.image for b in batch], dim=0)  # (B, C, H, W)
    masks = torch.stack([b.valid_mask for b in batch], dim=0)  # (B, 1, H, W)
    labels = torch.stack([b.label for b in batch], dim=0)  # (B,)
    metas: List[Dict[str, Any]] = [b.meta for b in batch]
    return ChipSample(image=images, valid_mask=masks, label=labels, meta=metas)

