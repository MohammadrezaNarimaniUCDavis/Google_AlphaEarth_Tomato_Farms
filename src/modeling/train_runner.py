"""Training loop for tomato U-Net on AlphaEarth chips."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modeling.dataset import AlphaEarthChipSegDataset, load_chips_table
from src.modeling.losses import combined_loss
from src.modeling.metrics import (
    binary_confusion_counts,
    chip_level_correct_counts,
    metrics_from_counts,
)
from src.modeling.model import TomatoUNet
from src.modeling.logging_utils import append_metrics_csv, write_json
from src.utils.paths import REPO_ROOT


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _print_device_banner(device: torch.device) -> None:
    print("=" * 60, flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        p = torch.cuda.get_device_properties(0)
        print(
            f"CUDA {p.major}.{p.minor}  Total VRAM: {p.total_memory / (1024**3):.2f} GiB",
            flush=True,
        )
    else:
        print("Warning: training on CPU — expect very slow runs.", flush=True)
    print("=" * 60, flush=True)


def _env_tqdm_disabled() -> bool:
    return os.environ.get("TRAINING_NO_TQDM", "").strip().lower() in ("1", "true", "yes")


def _pos_weight_from_df(df, split: str, device: torch.device) -> torch.Tensor:
    sub = df[df["split"] == split]
    n_t = (sub["class_label"].str.lower() == "tomato").sum()
    n_n = (sub["class_label"].str.lower() == "non_tomato").sum()
    n_n = max(int(n_n), 1)
    n_t = max(int(n_t), 1)
    w = float(n_n / n_t)
    return torch.tensor([w], device=device, dtype=torch.float32)


def _confusion_dict(tp: float, fp: float, fn: float, tn: float) -> dict[str, Any]:
    """2×2 confusion + readable labels (negative class = non-tomato, positive = tomato)."""
    return {
        "matrix_2x2": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "row_labels": ["true_non_tomato", "true_tomato"],
        "col_labels": ["pred_non_tomato", "pred_tomato"],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def _eval_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    bce_w: float,
    dice_w: float,
    *,
    desc: str = "eval",
    show_progress: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Micro-averaged pixel metrics over the whole split; chip-level accuracy; confusion JSON."""
    model.eval()
    tp = fp = fn = tn = 0.0
    chip_ok = chip_n = 0
    loss_sum = 0.0
    n_batches = 0
    it = loader
    if show_progress and not _env_tqdm_disabled():
        it = tqdm(
            loader,
            desc=desc,
            leave=False,
            mininterval=0.25,
            file=sys.stdout,
            dynamic_ncols=True,
        )
    for batch in it:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        m = batch["mask"].to(device)
        logits = model(x)
        loss = combined_loss(logits, y, m, bce_weight=bce_w, dice_weight=dice_w, pos_weight=pos_weight)
        loss_sum += float(loss.item())
        c = binary_confusion_counts(logits, y, m)
        tp += float(c["tp"].item())
        fp += float(c["fp"].item())
        fn += float(c["fn"].item())
        tn += float(c["tn"].item())
        co, cn = chip_level_correct_counts(logits, y, m)
        chip_ok += co
        chip_n += cn
        n_batches += 1
        if show_progress and not _env_tqdm_disabled() and isinstance(it, tqdm):
            it.set_postfix(loss=f"{float(loss.item()):.4f}")
    if n_batches == 0:
        z: dict[str, float] = {
            "loss": 0.0,
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "chip_acc": 0.0,
        }
        return z, _confusion_dict(0.0, 0.0, 0.0, 0.0)
    metrics = metrics_from_counts(tp, fp, fn, tn)
    metrics["loss"] = loss_sum / n_batches
    metrics["chip_acc"] = float(chip_ok / chip_n) if chip_n else 0.0
    conf = _confusion_dict(tp, fp, fn, tn)
    return metrics, conf


METRICS_EPOCH_FIELDS = [
    "epoch",
    "train_loss_opt",
    "train_loss",
    "train_acc",
    "train_precision",
    "train_recall",
    "train_f1",
    "train_iou",
    "train_chip_acc",
    "val_loss",
    "val_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_iou",
    "val_chip_acc",
]


def train_model(cfg: dict[str, Any], repo_root: Path | None = None) -> Path:
    """Run full train/val; optional test. Returns experiment output directory."""
    root = repo_root or REPO_ROOT

    data_cfg = cfg.get("data", {})
    chips_csv = root / str(data_cfg.get("chips_index_csv", "data/splits/chips_index.csv"))
    df = load_chips_table(chips_csv)

    out_root = root / str(cfg.get("output", {}).get("experiments_dir", "outputs/experiments"))
    run_id = str(cfg.get("output", {}).get("run_id") or "")
    if not run_id:
        from src.modeling.logging_utils import utc_run_id

        run_id = utc_run_id()
    exp_dir = out_root / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_json(exp_dir / "config_resolved.json", cfg)

    target_hw = tuple(cfg.get("data", {}).get("target_hw", [128, 128]))
    train_ds = AlphaEarthChipSegDataset(df, "train", target_hw, augment=True)
    val_ds = AlphaEarthChipSegDataset(df, "val", target_hw, augment=False)
    test_ds = AlphaEarthChipSegDataset(df, "test", target_hw, augment=False) if len(df[df["split"] == "test"]) else None

    bs = int(cfg.get("training", {}).get("batch_size", 8))
    nw = int(cfg.get("training", {}).get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=torch.cuda.is_available())
    train_eval_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())
    test_loader = (
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())
        if test_ds is not None
        else None
    )

    in_ch = int(cfg.get("model", {}).get("in_channels", 64))
    if cfg.get("model", {}).get("infer_in_channels", True):
        row0 = df.iloc[0]
        from src.modeling.io_paths import resolve_raster_path
        import pandas as pd

        lp = row0["local_path"]
        if "s3_uri" in row0.index and pd.notna(row0["s3_uri"]) and str(row0["s3_uri"]).strip().startswith("s3://"):
            su = str(row0["s3_uri"]).strip()
        else:
            su = None
        rpath = resolve_raster_path(lp, su)
        import rasterio

        with rasterio.open(rpath) as ds:
            in_ch = int(ds.count)

    device = _device()
    dropout_p = float(cfg.get("model", {}).get("dropout_p", 0.1))
    base = int(cfg.get("model", {}).get("base_channels", 32))
    model = TomatoUNet(in_channels=in_ch, base=base, dropout_p=dropout_p).to(device)

    lr = float(cfg.get("training", {}).get("learning_rate", 1e-3))
    wd = float(cfg.get("training", {}).get("weight_decay", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = int(cfg.get("training", {}).get("epochs", 30))
    bce_w = float(cfg.get("training", {}).get("bce_weight", 0.5))
    dice_w = float(cfg.get("training", {}).get("dice_weight", 0.5))
    use_pos_weight = bool(cfg.get("training", {}).get("use_pos_weight", True))
    pos_w = _pos_weight_from_df(df, "train", device) if use_pos_weight else None

    use_tqdm = bool(cfg.get("training", {}).get("tqdm", True)) and not _env_tqdm_disabled()
    _print_device_banner(device)
    print(
        f"Run: {run_id}  train chips={len(train_ds)}  val={len(val_ds)}  batch={bs}  epochs={epochs}",
        flush=True,
    )

    metrics_csv = exp_dir / "metrics_epoch.csv"
    best_val = -1.0
    best_path = exp_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_it: Any = train_loader
        if use_tqdm:
            train_it = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs} train",
                mininterval=0.25,
                file=sys.stdout,
                dynamic_ncols=True,
            )
        for batch_idx, batch in enumerate(train_it):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            m = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = combined_loss(logits, y, m, bce_weight=bce_w, dice_weight=dice_w, pos_weight=pos_w)
            loss.backward()
            opt.step()
            li = float(loss.item())
            train_losses.append(li)
            if use_tqdm and isinstance(train_it, tqdm):
                train_it.set_postfix(loss=f"{li:.4f}")
            if epoch == 1 and batch_idx == 0 and device.type == "cuda":
                torch.cuda.synchronize()
                print(
                    f"[GPU] After first training batch: allocated "
                    f"{torch.cuda.memory_allocated() / (1024**3):.3f} GiB  "
                    f"reserved {torch.cuda.memory_reserved() / (1024**3):.3f} GiB",
                    flush=True,
                )
        tr_loss = float(np.mean(train_losses)) if train_losses else 0.0

        print(f"Epoch {epoch}: running train-split eval (no gradients)…", flush=True)
        train_m, train_conf = _eval_split(
            model,
            train_eval_loader,
            device,
            pos_w,
            bce_w,
            dice_w,
            desc=f"Epoch {epoch} train [eval]",
            show_progress=use_tqdm,
        )
        print(f"Epoch {epoch}: running validation…", flush=True)
        val_m, val_conf = _eval_split(
            model,
            val_loader,
            device,
            pos_w,
            bce_w,
            dice_w,
            desc=f"Epoch {epoch} val",
            show_progress=use_tqdm,
        )
        row = {
            "epoch": epoch,
            "train_loss_opt": tr_loss,
            "train_loss": train_m["loss"],
            "train_acc": train_m["acc"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "train_f1": train_m["f1"],
            "train_iou": train_m["iou"],
            "train_chip_acc": train_m["chip_acc"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["acc"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "val_iou": val_m["iou"],
            "val_chip_acc": val_m["chip_acc"],
        }
        append_metrics_csv(metrics_csv, row, fieldnames=METRICS_EPOCH_FIELDS)
        write_json(exp_dir / "confusion_train_last.json", train_conf)
        write_json(exp_dir / "confusion_val_last.json", val_conf)
        print(
            f"Epoch {epoch}/{epochs} train_loss_opt={tr_loss:.4f} "
            f"train_iou={train_m['iou']:.4f} val_iou={val_m['iou']:.4f} val_f1={val_m['f1']:.4f}"
        )

        if val_m["iou"] > best_val:
            best_val = val_m["iou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, best_path)

        last_path = exp_dir / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, last_path)

    # Test
    if test_loader is not None and best_path.is_file():
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("Running test split…", flush=True)
        te, test_conf = _eval_split(
            model,
            test_loader,
            device,
            pos_w,
            bce_w,
            dice_w,
            desc="test",
            show_progress=use_tqdm,
        )
        write_json(exp_dir / "metrics_test.json", te)
        write_json(exp_dir / "confusion_test.json", test_conf)
        test_row = {
            "split": "test",
            "loss": te["loss"],
            "acc": te["acc"],
            "precision": te["precision"],
            "recall": te["recall"],
            "f1": te["f1"],
            "iou": te["iou"],
            "chip_acc": te["chip_acc"],
        }
        append_metrics_csv(exp_dir / "metrics_test_summary.csv", test_row)
        print("Test:", te)

    # Copy to SageMaker model dir if present
    sm = os.environ.get("SM_MODEL_DIR")
    if sm:
        d = Path(sm)
        d.mkdir(parents=True, exist_ok=True)
        if best_path.is_file():
            shutil.copy2(best_path, d / "model.pt")
        if metrics_csv.is_file():
            shutil.copy2(metrics_csv, d / "metrics_epoch.csv")

    return exp_dir
