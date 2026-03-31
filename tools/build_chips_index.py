"""Build train/val/test index for AlphaEarth clips (tomato vs non_tomato).

This script scans the local GeoTIFF chips under ``data/derived/alpha_earth_clips/ee``,
derives class labels from folder names, creates balanced splits, and writes
an index file to ``data/splits/chips_index.parquet`` (and CSV alongside it).

It does **not** move or modify any data; it just records paths + labels.
Run from the repo root:

    python tools/build_chips_index.py

Prereqs: pandas (see requirements.txt). Parquet uses pyarrow if available;
otherwise it will fall back to CSV only.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from src.utils.paths import REPO_ROOT, load_paths_config, resolve_under_root


def _find_chips(base: Path) -> List[Path]:
    """Return all .tif/.tiff files recursively under base (sorted for stability)."""
    if not base.is_dir():
        raise FileNotFoundError(f"AlphaEarth clips folder does not exist: {base}")
    exts = {".tif", ".tiff"}
    out: List[Path] = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


def _label_for_path(p: Path) -> str:
    """Map file path to class label (tomato vs non_tomato) based on folder name."""
    # Convention from existing runs: ee/landiq2018/... vs ee/landiq2018_non_tomato/...
    parts = [s.lower() for s in p.parts]
    for s in parts:
        if "non_tomato" in s:
            return "non_tomato"
    return "tomato"


def _balanced_splits(
    tomato: List[int],
    non_tomato: List[int],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 1337,
) -> Tuple[List[int], List[int], List[int]]:
    """Return index lists for train, val, test with equal tomato/non_tomato counts.

    We assume tomato + non_tomato are disjoint index sets into the same chips list.
    """
    rng = random.Random(seed)
    rng.shuffle(tomato)
    rng.shuffle(non_tomato)

    # Use only the min class count to keep splits perfectly balanced.
    n_per_class = min(len(tomato), len(non_tomato))
    n_train = int(train_frac * n_per_class)
    n_val = int(val_frac * n_per_class)
    n_test = n_per_class - n_train - n_val

    tomato_use = tomato[:n_per_class]
    non_use = non_tomato[:n_per_class]

    def take_slices(idxs: List[int]) -> Tuple[List[int], List[int], List[int]]:
        tr = idxs[:n_train]
        va = idxs[n_train : n_train + n_val]
        te = idxs[n_train + n_val : n_train + n_val + n_test]
        return tr, va, te

    t_tr, t_va, t_te = take_slices(tomato_use)
    n_tr, n_va, n_te = take_slices(non_use)

    train = sorted(t_tr + n_tr)
    val = sorted(t_va + n_va)
    test = sorted(t_te + n_te)
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser(description="Build chips_index for AlphaEarth tomato vs non_tomato.")
    ap.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for shuffling before splits (default: 1337).",
    )
    args = ap.parse_args()

    cfg = load_paths_config()
    data_cfg = cfg.get("data", {})
    clips_rel = data_cfg.get("alpha_earth_clips", "data/derived/alpha_earth_clips")
    clips_dir = resolve_under_root(str(clips_rel), REPO_ROOT)

    ee_dir = clips_dir / "ee"
    chips = _find_chips(ee_dir)
    if not chips:
        raise SystemExit(f"No .tif/.tiff files found under {ee_dir}")

    # Build DataFrame with one row per chip.
    rows = []
    for i, p in enumerate(chips):
        label = _label_for_path(p)
        rel = p.relative_to(REPO_ROOT)
        rows.append(
            {
                "chip_index": i,
                "chip_id": p.stem,
                "class_label": label,
                "local_path": str(rel).replace("\\", "/"),
            }
        )
    df = pd.DataFrame(rows)

    # Determine S3 URI if bucket is configured.
    from src.utils import s3_layout  # imported late to avoid circulars

    bucket = s3_layout.s3_bucket(cfg)
    if bucket:
        # Build URIs using s3_layout + local path inside alpha_earth_clips root.
        clips_root_rel = Path(clips_rel)
        if clips_root_rel.is_absolute():
            # Approximate: strip REPO_ROOT if user used absolute paths in config.
            try:
                clips_root_rel = clips_dir.relative_to(REPO_ROOT)
            except ValueError:
                clips_root_rel = Path("data/derived/alpha_earth_clips")
        project_prefix = s3_layout.s3_project_root_prefix(cfg)
        alpha_key = cfg.get("s3", {}).get("keys", {}).get("alpha_earth_clips", "derived/alpha_earth_clips")

        def to_s3_uri(local_path: str) -> str:
            lp = Path(local_path)
            try:
                rel_inside = lp.relative_to(clips_root_rel)
            except ValueError:
                rel_inside = lp
            key = "/".join(
                [
                    project_prefix.strip("/"),
                    alpha_key.strip("/"),
                    str(rel_inside).replace("\\", "/"),
                ]
            )
            return f"s3://{bucket}/{key}"

        df["s3_uri"] = df["local_path"].map(to_s3_uri)

    # Build balanced splits.
    tomato_idx = df.index[df["class_label"] == "tomato"].tolist()
    non_idx = df.index[df["class_label"] == "non_tomato"].tolist()
    train_idx, val_idx, test_idx = _balanced_splits(tomato_idx, non_idx, seed=args.seed)

    split = pd.Series(index=df.index, dtype="object")
    split.loc[train_idx] = "train"
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"
    df["split"] = split

    # Drop any rows without a split (if class counts were unequal).
    before = len(df)
    df = df[df["split"].notna()].copy()

    # Write outputs.
    splits_dir = resolve_under_root(cfg.get("data", {}).get("splits", "data/splits"), REPO_ROOT)
    splits_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = splits_dir / "chips_index.parquet"
    csv_path = splits_dir / "chips_index.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        wrote_parquet = True
    except Exception:
        wrote_parquet = False
    df.to_csv(csv_path, index=False)

    # Simple summary.
    summary = (
        df.groupby(["split", "class_label"])["chip_id"]
        .count()
        .rename("count")
        .reset_index()
        .sort_values(["split", "class_label"])
    )

    print(f"Wrote index with {len(df)} chips (dropped {before - len(df)} to balance classes).")
    print(f"CSV:     {csv_path}")
    if wrote_parquet:
        print(f"Parquet: {parquet_path}")
    else:
        print("Parquet: (failed to write; pyarrow/fastparquet may be missing, CSV still written).")
    print("\nCounts by split and class_label:\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

