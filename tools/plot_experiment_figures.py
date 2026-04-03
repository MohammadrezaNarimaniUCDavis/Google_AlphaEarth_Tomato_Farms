#!/usr/bin/env python3
"""Build paper-style figures from a SageMaker/local experiment directory.

Reads ``metrics_epoch.csv``, ``confusion_test.json`` (or ``--confusion-split``), and
``metrics_test.json`` if present. Writes PNG (high DPI) into ``--out-dir``.

Example::

    pip install -e ".[figures]"
    python tools/plot_experiment_figures.py \\
      --experiment-dir archive/sagemaker-studio/project-outputs/experiments/20260402T230522Z \\
      --out-dir figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib\n"
            "or: pip install -e \".[figures]\""
        ) from e
    return plt


def plot_learning_curves(csv_path: Path, out_png: Path, plt) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    epochs = df["epoch"].values

    ax = axes[0, 0]
    ax.plot(epochs, df["train_loss_opt"], label="train (opt)", lw=1.5)
    ax.plot(epochs, df["val_loss"], label="val", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.set_title("Loss")

    ax = axes[0, 1]
    ax.plot(epochs, df["train_iou"], label="train IoU", lw=1.5)
    ax.plot(epochs, df["val_iou"], label="val IoU", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.legend(frameon=False)
    ax.set_title("IoU")

    ax = axes[1, 0]
    ax.plot(epochs, df["train_f1"], label="train F1", lw=1.5)
    ax.plot(epochs, df["val_f1"], label="val F1", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.legend(frameon=False)
    ax.set_title("F1")

    ax = axes[1, 1]
    ax.plot(epochs, df["train_chip_acc"], label="train chip acc", lw=1.5)
    ax.plot(epochs, df["val_chip_acc"], label="val chip acc", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Chip accuracy")
    ax.legend(frameon=False)
    ax.set_title("Chip accuracy")

    fig.suptitle("Training curves", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_confusion_2x2(path: Path) -> tuple[list[list[float]], list[str], list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mat = data["matrix_2x2"]
    rows = data["row_labels"]
    cols = data["col_labels"]
    arr = [[float(mat[i][j]) for j in range(2)] for i in range(2)]
    return arr, rows, cols


def plot_confusion(
    path: Path,
    out_png: Path,
    plt,
    *,
    mode: str = "global",
    title_suffix: str = "",
) -> None:
    """mode: global = % of all pixels; row = each row sums to 1; col = each column sums to 1."""
    arr, rows, cols = _load_confusion_2x2(path)

    if mode == "global":
        total = sum(sum(r) for r in arr)
        disp = [[arr[i][j] / total for j in range(2)] for i in range(2)]
        vmin, vmax = 0.0, 1.0
        cbar_label = "Fraction of all pixels"
        title = f"Confusion (global){title_suffix}"
        fmt_cell = lambda i, j: f"{disp[i][j] * 100:.2f}%\n(n={int(arr[i][j])})"
    elif mode == "row":
        row_sums = [sum(arr[i][j] for j in range(2)) for i in range(2)]
        disp = [[arr[i][j] / row_sums[i] for j in range(2)] for i in range(2)]
        vmin, vmax = 0.0, 1.0
        cbar_label = "Fraction (each row sums to 1)"
        title = f"Confusion row-normalized — P(pred | true){title_suffix}"
        fmt_cell = lambda i, j: f"{disp[i][j]:.4f}\n(n={int(arr[i][j])})"
    elif mode == "col":
        col_sums = [sum(arr[i][j] for i in range(2)) for j in range(2)]
        disp = [[arr[i][j] / col_sums[j] for j in range(2)] for i in range(2)]
        vmin, vmax = 0.0, 1.0
        cbar_label = "Fraction (each column sums to 1)"
        title = f"Confusion column-normalized — P(true | pred){title_suffix}"
        fmt_cell = lambda i, j: f"{disp[i][j]:.4f}\n(n={int(arr[i][j])})"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    im = ax.imshow(disp, cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticklabels(rows)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, fmt_cell(i, j), ha="center", va="center", color="black", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(path: Path, out_png: Path, plt) -> None:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    keys = ["acc", "precision", "recall", "f1", "iou", "chip_acc"]
    labels = ["Pixel acc", "Precision", "Recall", "F1", "IoU", "Chip acc"]
    vals = [float(m[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = range(len(labels))
    ax.bar(x, vals, color="#2c5282")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test set metrics")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot learning curves and confusion from experiment outputs")
    ap.add_argument(
        "--experiment-dir",
        type=Path,
        default=_REPO / "archive/sagemaker-studio/project-outputs/experiments/20260402T230522Z",
        help="Directory containing metrics_epoch.csv and confusion_*.json",
    )
    ap.add_argument("--out-dir", type=Path, default=_REPO / "figures", help="Where to write PNGs")
    ap.add_argument(
        "--confusion-split",
        type=str,
        default="test",
        choices=("test", "val"),
        help="Which confusion_*.json to plot if present",
    )
    args = ap.parse_args()

    exp = args.experiment_dir
    if not exp.is_dir():
        raise SystemExit(f"Not a directory: {exp}")

    plt = _require_matplotlib()

    out = args.out_dir
    csv_path = exp / "metrics_epoch.csv"
    if csv_path.is_file():
        plot_learning_curves(csv_path, out / "learning_curves.png", plt)
        print(f"Wrote {out / 'learning_curves.png'}")
    else:
        print(f"Skip learning curves (missing {csv_path})")

    conf_name = f"confusion_{args.confusion_split}.json"
    conf_path = exp / conf_name
    if conf_path.is_file():
        split = args.confusion_split
        plot_confusion(conf_path, out / f"confusion_matrix_{split}.png", plt, mode="global")
        print(f"Wrote {out / f'confusion_matrix_{split}.png'}")
        plot_confusion(
            conf_path,
            out / f"confusion_matrix_{split}_row_normalized.png",
            plt,
            mode="row",
        )
        print(f"Wrote {out / f'confusion_matrix_{split}_row_normalized.png'}")
        plot_confusion(
            conf_path,
            out / f"confusion_matrix_{split}_col_normalized.png",
            plt,
            mode="col",
        )
        print(f"Wrote {out / f'confusion_matrix_{split}_col_normalized.png'}")
    else:
        print(f"Skip confusion (missing {conf_path})")

    test_metrics = exp / "metrics_test.json"
    if test_metrics.is_file():
        plot_metrics_bar(test_metrics, out / "metrics_bar_test.png", plt)
        print(f"Wrote {out / 'metrics_bar_test.png'}")
    else:
        print(f"Skip metrics bar (missing {test_metrics})")


if __name__ == "__main__":
    main()
