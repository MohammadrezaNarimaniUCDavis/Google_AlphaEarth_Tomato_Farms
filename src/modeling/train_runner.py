"""Training loop for tomato U-Net on AlphaEarth chips."""

from __future__ import annotations

import contextlib
import csv
import itertools
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
from src.modeling.io_paths import resolve_raster_path
from src.modeling.losses import combined_loss
from src.modeling.metrics import (
    binary_confusion_counts,
    chip_level_correct_counts,
    metrics_from_counts,
)
from src.modeling.model import TomatoUNet
from src.modeling.logging_utils import (
    append_metrics_csv,
    maybe_sync_experiment_to_s3,
    utc_iso,
    utc_run_id,
    write_experiment_complete,
    write_json,
    write_run_manifest,
)
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


def _apply_training_env_overrides(cfg: dict[str, Any]) -> None:
    """Override ``training.*`` from env (helps resume after OOM with fewer workers)."""
    tr = cfg.setdefault("training", {})
    changed: list[str] = []
    if os.environ.get("ALPHA_EARTH_TRAIN_FAST", "").strip().lower() in ("1", "true", "yes"):
        tr.update(
            {
                "use_amp": True,
                "num_workers": 4,
                "batch_size": 24,
                "prefetch_factor": 8,
            }
        )
        print(
            "[config] ALPHA_EARTH_TRAIN_FAST=1 preset: workers=4 batch=24 prefetch=8 use_amp=True "
            "(train workers may be capped for S3 /vsis3/)",
            flush=True,
        )
    for key, envname in (
        ("num_workers", "ALPHA_EARTH_TRAIN_NUM_WORKERS"),
        ("eval_num_workers", "ALPHA_EARTH_TRAIN_EVAL_NUM_WORKERS"),
        ("prefetch_factor", "ALPHA_EARTH_TRAIN_PREFETCH_FACTOR"),
        ("batch_size", "ALPHA_EARTH_TRAIN_BATCH_SIZE"),
    ):
        raw = os.environ.get(envname, "").strip()
        if not raw:
            continue
        try:
            tr[key] = int(raw)
            changed.append(f"{key}={tr[key]}")
        except ValueError:
            pass
    amp_raw = os.environ.get("ALPHA_EARTH_TRAIN_USE_AMP", "").strip().lower()
    if amp_raw in ("0", "false", "no", "off"):
        tr["use_amp"] = False
        changed.append("use_amp=False")
    elif amp_raw in ("1", "true", "yes", "on"):
        tr["use_amp"] = True
        changed.append("use_amp=True")
    if changed:
        print(f"[config] Env overrides applied: {', '.join(changed)}", flush=True)


def _autocast(device: torch.device, use_amp: bool):
    if device.type == "cuda" and use_amp:
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def _train_num_workers_capped_for_s3(
    nw: int,
    train_ds: AlphaEarthChipSegDataset,
    cfg: dict[str, Any],
) -> int:
    """Forked DataLoader workers + GDAL S3 reads multiply RAM; OOM often kills the job with no traceback."""
    import pandas as pd

    row = train_ds.df.iloc[0]
    lp = row["local_path"]
    if "s3_uri" in row.index and pd.notna(row["s3_uri"]) and str(row["s3_uri"]).strip().startswith("s3://"):
        su = str(row["s3_uri"]).strip()
    else:
        su = None
    rpath = resolve_raster_path(lp, su)
    if not str(rpath).startswith("/vsis3/"):
        return nw
    cap_raw = os.environ.get("ALPHA_EARTH_S3_TRAIN_WORKERS_CAP", "").strip()
    if cap_raw != "":
        try:
            cap = max(0, int(cap_raw))
        except ValueError:
            cap = 2
    else:
        c = cfg.get("training", {}).get("s3_train_num_workers_cap")
        cap = 2 if c is None else max(0, int(c))
    out = min(nw, cap)
    if out < nw:
        print(
            f"[config] S3 train I/O (/vsis3/): num_workers {nw} -> {out} "
            f"(s3_train_num_workers_cap={cap}; env ALPHA_EARTH_S3_TRAIN_WORKERS_CAP overrides)",
            flush=True,
        )
    return out


def _val_test_num_workers(
    split_ds: AlphaEarthChipSegDataset | None,
    eval_nw: int,
    nw_train: int,
    cfg: dict[str, Any],
    *,
    split_label: str,
) -> int:
    """If chips resolve to **local files**, use ``nw_train`` for val/test loaders (fast). /vsis3/ keeps ``eval_nw``."""
    force = os.environ.get("ALPHA_EARTH_FORCE_VAL_TEST_WORKERS", "").strip()
    if force != "":
        try:
            return max(0, int(force))
        except ValueError:
            pass
    if split_ds is None or len(split_ds.df) == 0:
        return eval_nw
    if not bool(cfg.get("training", {}).get("eval_fast_if_local", True)):
        return eval_nw
    import pandas as pd

    row = split_ds.df.iloc[0]
    lp = row["local_path"]
    if "s3_uri" in row.index and pd.notna(row["s3_uri"]) and str(row["s3_uri"]).strip().startswith("s3://"):
        su = str(row["s3_uri"]).strip()
    else:
        su = None
    try:
        rpath = resolve_raster_path(lp, su)
    except (FileNotFoundError, OSError, ValueError):
        return eval_nw
    if str(rpath).startswith("/vsis3/"):
        return eval_nw
    out = nw_train
    cap = cfg.get("training", {}).get("eval_local_max_workers")
    if cap is not None:
        out = min(out, max(0, int(cap)))
    if out != eval_nw:
        print(
            f"[config] {split_label} on local disk → DataLoader workers={out} "
            f"(eval_num_workers={eval_nw} in YAML is S3-only fallback)",
            flush=True,
        )
    return out


def _dataloader_kwargs(num_workers: int, prefetch_factor: int | None) -> dict[str, Any]:
    """Pin memory + optional parallel prefetch (num_workers>0 speeds S3/disk I/O vs GPU)."""
    kw: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        pf = int(prefetch_factor) if prefetch_factor is not None else 4
        kw["prefetch_factor"] = max(2, pf)
    return kw


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
    max_batches: int | None = None,
    use_amp: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Micro-averaged pixel metrics over the whole split; chip-level accuracy; confusion JSON."""
    model.eval()
    tp = fp = fn = tn = 0.0
    chip_ok = chip_n = 0
    loss_sum = 0.0
    n_batches = 0
    cap = max_batches if max_batches is not None else len(loader)
    loader_iter: Any = itertools.islice(loader, cap) if max_batches is not None else loader
    if show_progress and not _env_tqdm_disabled():
        it = tqdm(
            loader_iter,
            desc=desc,
            total=min(cap, len(loader)) if max_batches is not None else len(loader),
            leave=False,
            mininterval=0.25,
            file=sys.stdout,
            dynamic_ncols=True,
        )
    else:
        it = loader_iter
    nb = device.type == "cuda"
    for batch in it:
        x = batch["x"].to(device, non_blocking=nb)
        y = batch["y"].to(device, non_blocking=nb)
        m = batch["mask"].to(device, non_blocking=nb)
        with _autocast(device, use_amp):
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


def train_model(
    cfg: dict[str, Any],
    repo_root: Path | None = None,
    *,
    resume_from: Path | None = None,
    command_argv: list[str] | None = None,
) -> Path:
    """Run full train/val; optional test. Returns experiment output directory.

    If ``resume_from`` points to a checkpoint (e.g. ``last.pt``) under
    ``outputs/experiments/<run_id>/``, training continues from the next epoch
    using ``cfg`` stored in that checkpoint (YAML overrides are ignored for
    resumed keys — pass the same config file as the original run).
    """
    root = repo_root or REPO_ROOT
    resume_from = Path(resume_from).resolve() if resume_from else None
    ck_resume: dict[str, Any] | None = None
    if resume_from is not None:
        if not resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        try:
            ck_resume = torch.load(resume_from, map_location="cpu", weights_only=False)
        except TypeError:
            ck_resume = torch.load(resume_from, map_location="cpu")
        cfg = ck_resume["cfg"]

    data_cfg = cfg.get("data", {})
    chips_csv = root / str(data_cfg.get("chips_index_csv", "data/splits/chips_index.csv"))
    df = load_chips_table(chips_csv)

    out_root = root / str(cfg.get("output", {}).get("experiments_dir", "outputs/experiments"))
    run_id = str(cfg.get("output", {}).get("run_id") or "")
    if resume_from is not None:
        exp_dir = resume_from.parent
        run_id = exp_dir.name
    else:
        if not run_id:
            run_id = utc_run_id()
        exp_dir = out_root / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    _apply_training_env_overrides(cfg)
    write_json(exp_dir / "config_resolved.json", cfg)
    if resume_from is None:
        write_run_manifest(exp_dir, cfg, repo_root=root, command_argv=command_argv)
    else:
        write_json(
            exp_dir / f"resume_{utc_run_id()}.json",
            {
                "resumed_utc": utc_iso(),
                "resume_checkpoint": str(resume_from),
                "argv": command_argv or sys.argv,
            },
        )

    target_hw = tuple(cfg.get("data", {}).get("target_hw", [128, 128]))
    train_ds = AlphaEarthChipSegDataset(df, "train", target_hw, augment=True)
    val_ds = AlphaEarthChipSegDataset(df, "val", target_hw, augment=False)
    test_ds = AlphaEarthChipSegDataset(df, "test", target_hw, augment=False) if len(df[df["split"] == "test"]) else None

    bs = int(cfg.get("training", {}).get("batch_size", 8))
    nw = int(cfg.get("training", {}).get("num_workers", 0))
    # S3 + rasterio in forked workers caused bogus ObjectNotFound on **val** (different URIs).
    # Train-split eval uses the same chips as training → reuse train workers + prefetch (much faster).
    eval_nw = int(cfg.get("training", {}).get("eval_num_workers", 0))
    pf_raw = cfg.get("training", {}).get("prefetch_factor")
    prefetch_factor: int | None = int(pf_raw) if pf_raw is not None else None
    nw_train = _train_num_workers_capped_for_s3(nw, train_ds, cfg)
    dl_train_kw = _dataloader_kwargs(nw_train, prefetch_factor)
    val_nw = _val_test_num_workers(val_ds, eval_nw, nw_train, cfg, split_label="val")
    test_nw = (
        _val_test_num_workers(test_ds, eval_nw, nw_train, cfg, split_label="test")
        if test_ds is not None
        else eval_nw
    )
    dl_val_kw = _dataloader_kwargs(val_nw, prefetch_factor)
    dl_test_kw = _dataloader_kwargs(test_nw, prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **dl_train_kw)
    train_eval_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, **dl_train_kw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **dl_val_kw)
    test_loader = (
        DataLoader(test_ds, batch_size=bs, shuffle=False, **dl_test_kw) if test_ds is not None else None
    )

    in_ch = int(cfg.get("model", {}).get("in_channels", 64))
    if cfg.get("model", {}).get("infer_in_channels", True):
        row0 = df.iloc[0]
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
    use_amp = bool(cfg.get("training", {}).get("use_amp", True)) and device.type == "cuda"
    scaler: torch.amp.GradScaler | None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = None

    lr = float(cfg.get("training", {}).get("learning_rate", 1e-3))
    wd = float(cfg.get("training", {}).get("weight_decay", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = int(cfg.get("training", {}).get("epochs", 30))
    bce_w = float(cfg.get("training", {}).get("bce_weight", 0.5))
    dice_w = float(cfg.get("training", {}).get("dice_weight", 0.5))
    use_pos_weight = bool(cfg.get("training", {}).get("use_pos_weight", True))
    pos_w = _pos_weight_from_df(df, "train", device) if use_pos_weight else None

    use_tqdm = bool(cfg.get("training", {}).get("tqdm", True)) and not _env_tqdm_disabled()

    def _opt_cap(key: str) -> int | None:
        v = cfg.get("training", {}).get(key)
        if v is None or v == "":
            return None
        n = int(v)
        return n if n > 0 else None

    max_train_batches = _opt_cap("max_train_batches")
    max_eval_batches = _opt_cap("max_eval_batches")

    _print_device_banner(device)
    print(
        f"Run: {run_id}  train chips={len(train_ds)}  val={len(val_ds)}  batch={bs}  epochs={epochs}  "
        f"workers={nw_train}  val_test_workers=({val_nw},{test_nw})  eval_num_workers_yaml={eval_nw}  "
        f"AMP={'on' if use_amp else 'off'}  "
        f"cudnn.benchmark={device.type == 'cuda'}",
        flush=True,
    )
    if max_train_batches or max_eval_batches:
        print(
            f"SMOKE / SUBSET: max_train_batches={max_train_batches} max_eval_batches={max_eval_batches} "
            f"(metrics are on partial data only — use full config for real training)",
            flush=True,
        )

    metrics_csv = exp_dir / "metrics_epoch.csv"
    best_path = exp_dir / "best.pt"
    start_epoch = 1
    best_val = -1.0
    last_row: dict[str, Any] | None = None

    if ck_resume is not None:
        model.load_state_dict(ck_resume["model"])
        start_epoch = int(ck_resume.get("epoch", 0)) + 1
        if metrics_csv.is_file():
            with metrics_csv.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        vi = float(r.get("val_iou", "nan"))
                        if vi == vi and vi > best_val:
                            best_val = vi
                    except (TypeError, ValueError):
                        pass
        print(
            f"Resuming from {resume_from.name}: completed_epoch={start_epoch - 1} "
            f"-> epochs {start_epoch}..{epochs}  best_val_iou_from_csv={best_val:.4f}",
            flush=True,
        )

    if start_epoch > epochs:
        print(
            f"Training already complete (checkpoint epoch >= config epochs={epochs}). "
            f"Skipping training loop; running test/finalize if applicable.",
            flush=True,
        )
        if metrics_csv.is_file():
            with metrics_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last_row = rows[-1]

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_losses = []
        if max_train_batches is not None:
            train_iter: Any = itertools.islice(train_loader, max_train_batches)
            train_total = min(max_train_batches, len(train_loader))
        else:
            train_iter = train_loader
            train_total = len(train_loader)
        train_it: Any = train_iter
        if use_tqdm:
            train_it = tqdm(
                train_iter,
                desc=f"Epoch {epoch}/{epochs} train",
                total=train_total,
                mininterval=0.25,
                file=sys.stdout,
                dynamic_ncols=True,
            )
        nb = device.type == "cuda"
        for batch_idx, batch in enumerate(train_it):
            x = batch["x"].to(device, non_blocking=nb)
            y = batch["y"].to(device, non_blocking=nb)
            m = batch["mask"].to(device, non_blocking=nb)
            opt.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits = model(x)
                loss = combined_loss(logits, y, m, bce_weight=bce_w, dice_weight=dice_w, pos_weight=pos_w)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            li = float(loss.item())
            train_losses.append(li)
            if use_tqdm and isinstance(train_it, tqdm):
                train_it.set_postfix(loss=f"{li:.4f}")
            if epoch == start_epoch and batch_idx == 0 and device.type == "cuda":
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
            max_batches=max_eval_batches,
            use_amp=use_amp,
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
            max_batches=max_eval_batches,
            use_amp=use_amp,
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
        last_row = row
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
    metrics_test_out: dict[str, Any] | None = None
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
            max_batches=max_eval_batches,
            use_amp=use_amp,
        )
        metrics_test_out = dict(te)
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

    if last_row is None and metrics_csv.is_file():
        with metrics_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last_row = rows[-1]

    write_experiment_complete(
        exp_dir,
        metrics_test=metrics_test_out,
        last_val_row=last_row,
        best_val_iou=float(best_val),
    )
    sync_dest = maybe_sync_experiment_to_s3(exp_dir)
    if sync_dest:
        print(f"S3 archive: {sync_dest}", flush=True)

    # Copy to SageMaker model dir if present
    sm = os.environ.get("SM_MODEL_DIR")
    if sm:
        d = Path(sm)
        d.mkdir(parents=True, exist_ok=True)
        if best_path.is_file():
            shutil.copy2(best_path, d / "model.pt")
        if metrics_csv.is_file():
            shutil.copy2(metrics_csv, d / "metrics_epoch.csv")
        for name in ("metrics_test.json", "experiment_complete.json", "artifacts_index.json", "run_manifest.json"):
            p = exp_dir / name
            if p.is_file():
                shutil.copy2(p, d / name)

    return exp_dir
