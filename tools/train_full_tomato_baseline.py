"""Train a baseline AlphaEarth tomato vs non_tomato model on full dataset.

This script is meant to be run either locally (CPU/small GPU) or on the
SageMaker notebook instance. It reuses:

- `data/splits/chips_index.(parquet|csv)` built by `tools/build_chips_index.py`
- `AlphaEarthChipsDataset` + `collate_chips`
- `AlphaEarthTomatoModel`

Logging:
- Metrics and config are written under `logs/experiments/<run_name>/`.
- The final model checkpoint is saved as `model_final.pt` in that folder.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

from src.modeling.alpha_earth_dataset import AlphaEarthChipsDataset, collate_chips
from src.modeling.alpha_earth_model import AlphaEarthTomatoModel
from src.utils.paths import REPO_ROOT, load_paths_config


@dataclass
class TrainConfig:
    index_path: Path
    run_name: str = "baseline"
    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-3
    lambda_chip: float = 1.0
    lambda_pixel: float = 0.2
    subset_train: Optional[int] = None  # e.g. 10_000 for quick runs
    subset_val: Optional[int] = None
    num_workers: int = 0


def make_experiment_dir(run_name: str) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%S")
    root = REPO_ROOT / "logs" / "experiments" / f"{ts}_{run_name}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_dataloaders(cfg: TrainConfig) -> Dict[str, DataLoader]:
    train_ds_full = AlphaEarthChipsDataset(index_path=cfg.index_path, split="train")
    val_ds_full = AlphaEarthChipsDataset(index_path=cfg.index_path, split="val")

    if cfg.subset_train is not None:
        train_indices = list(range(min(cfg.subset_train, len(train_ds_full))))
        train_ds = Subset(train_ds_full, train_indices)
    else:
        train_ds = train_ds_full

    if cfg.subset_val is not None:
        val_indices = list(range(min(cfg.subset_val, len(val_ds_full))))
        val_ds = Subset(val_ds_full, val_indices)
    else:
        val_ds = val_ds_full

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_chips,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_chips,
    )
    return {"train": train_loader, "val": val_loader}


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary accuracy from logits and integer labels {0,1}."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    correct = (preds == labels.long()).float().mean().item()
    return float(correct)


def train_one_epoch(
    model: AlphaEarthTomatoModel,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    lambda_chip: float,
    lambda_pixel: float,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    bce_logits = torch.nn.BCEWithLogitsLoss()

    metrics: Dict[str, Any] = {}

    for phase in ("train", "val"):
        loader = loaders[phase]
        dataset_size = len(loader.dataset)
        n_total_batches = len(loader)
        print(f"  {phase}: {dataset_size} samples, {n_total_batches} batches")

        if phase == "train":
            model.train(True)
        else:
            model.train(False)

        running_loss = 0.0
        running_chip_acc = 0.0
        n_batches = 0

        # Print progress roughly 10 times per epoch per phase.
        print_every = max(1, n_total_batches // 10)

        for batch_idx, batch in enumerate(loader, start=1):
            x = batch.image.to(device)
            mask = batch.valid_mask.to(device)
            y = batch.label.to(device).float()  # (B,)

            if phase == "train":
                optimizer.zero_grad()

            pixel_logits, chip_logits = model(x)

            chip_loss = bce_logits(chip_logits, y)

            y_px = y.view(-1, 1, 1, 1).expand_as(pixel_logits)
            pixel_loss_raw = bce_logits(pixel_logits, y_px)
            valid_frac = mask.float().mean().clamp(min=1e-3)
            pixel_loss = pixel_loss_raw * valid_frac

            loss = lambda_chip * chip_loss + lambda_pixel * pixel_loss

            if phase == "train":
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                acc = accuracy_from_logits(chip_logits, y)

            running_loss += loss.item()
            running_chip_acc += acc
            n_batches += 1

            if batch_idx == 1 or batch_idx % print_every == 0 or batch_idx == n_total_batches:
                print(
                    f"    [{phase}] batch {batch_idx}/{n_total_batches} "
                    f"loss={loss.item():.4f} chip_acc={acc:.3f}"
                )

        avg_loss = running_loss / max(1, n_batches)
        avg_acc = running_chip_acc / max(1, n_batches)
        metrics[f"{phase}_loss"] = avg_loss
        metrics[f"{phase}_chip_acc"] = avg_acc

    return metrics


def resolve_index_path(arg_index: Optional[str]) -> Path:
    if arg_index is not None:
        p = Path(arg_index)
        if not p.is_file():
            raise FileNotFoundError(f"Provided index_path does not exist: {p}")
        return p

    cfg = load_paths_config()
    splits_dir = REPO_ROOT / cfg.get("data", {}).get("splits", "data/splits")
    parquet = splits_dir / "chips_index.parquet"
    csv = splits_dir / "chips_index.csv"
    if parquet.is_file():
        return parquet
    if csv.is_file():
        return csv
    raise FileNotFoundError(f"No chips_index found under {splits_dir}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train baseline tomato vs non_tomato model.")
    parser.add_argument("--index-path", type=str, default=None, help="Override path to chips_index parquet/csv.")
    parser.add_argument("--run-name", type=str, default="baseline", help="Short name for this run.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-chip", type=float, default=1.0)
    parser.add_argument("--lambda-pixel", type=float, default=0.2)
    parser.add_argument("--subset-train", type=int, default=None, help="If set, limit number of training examples.")
    parser.add_argument("--subset-val", type=int, default=None, help="If set, limit number of validation examples.")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args(argv)

    index_path = resolve_index_path(args.index_path)

    train_cfg = TrainConfig(
        index_path=index_path,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_chip=args.lambda_chip,
        lambda_pixel=args.lambda_pixel,
        subset_train=args.subset_train,
        subset_val=args.subset_val,
        num_workers=args.num_workers,
    )

    exp_dir = make_experiment_dir(train_cfg.run_name)
    print(f"Experiment dir: {exp_dir}")
    print(f"Using index: {train_cfg.index_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    loaders = build_dataloaders(train_cfg)

    model = AlphaEarthTomatoModel(in_channels=64, base_channels=32, dropout_p=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    all_metrics: list[Dict[str, Any]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        print(f"Epoch {epoch}/{train_cfg.epochs}")
        metrics = train_one_epoch(
            model=model,
            loaders=loaders,
            device=device,
            lambda_chip=train_cfg.lambda_chip,
            lambda_pixel=train_cfg.lambda_pixel,
            optimizer=optimizer,
        )
        metrics["epoch"] = epoch
        all_metrics.append(metrics)
        print(
            f"  train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"train_acc={metrics['train_chip_acc']:.3f} "
            f"val_acc={metrics['val_chip_acc']:.3f}"
        )

    # Save metrics and config.
    (exp_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (exp_dir / "train_config.json").write_text(json.dumps(asdict(train_cfg), indent=2, default=str))

    # Save final model weights.
    ckpt_path = exp_dir / "model_final.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"Saved final model checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()

