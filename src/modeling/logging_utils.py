"""CSV / JSONL run logs under outputs/experiments/<run_id>."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Git commit, Python/torch versions — for reproducibility and audit."""
    repo_root = repo_root or Path.cwd()
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "alpha_earth_data_source": os.environ.get("ALPHA_EARTH_DATA_SOURCE", ""),
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as e:
        out["torch_error"] = str(e)
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode == 0:
            out["git_commit"] = r.stdout.strip()
        r2 = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r2.returncode == 0 and r2.stdout.strip():
            out["git_dirty"] = True
    except Exception as e:
        out["git_error"] = str(e)
    return out


def write_run_manifest(
    exp_dir: Path,
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    command_argv: list[str] | None = None,
) -> None:
    """Written at **start** of training."""
    write_json(
        exp_dir / "run_manifest.json",
        {
            "started_utc": utc_iso(),
            "run_id": exp_dir.name,
            "config": cfg,
            "provenance": collect_provenance(repo_root),
            "argv": command_argv or sys.argv,
        },
    )


def list_artifact_files(exp_dir: Path) -> list[dict[str, Any]]:
    """Relative paths and sizes (bytes) for every file under ``exp_dir``."""
    rows: list[dict[str, Any]] = []
    for p in sorted(exp_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(exp_dir))
            rows.append({"path": rel, "bytes": p.stat().st_size})
    return rows


def write_experiment_complete(
    exp_dir: Path,
    *,
    metrics_test: dict[str, Any] | None,
    last_val_row: dict[str, Any] | None,
    best_val_iou: float,
) -> None:
    """Single JSON for papers / reuse: test + last-epoch val + pointers to files."""
    write_json(
        exp_dir / "experiment_complete.json",
        {
            "finished_utc": utc_iso(),
            "best_val_iou": best_val_iou,
            "metrics_test": metrics_test,
            "last_epoch_val_train_logged": last_val_row,
            "files": {
                "checkpoints": ["best.pt", "last.pt"],
                "metrics_epoch_csv": "metrics_epoch.csv",
                "confusion_test": "confusion_test.json",
                "metrics_test": "metrics_test.json",
                "run_manifest": "run_manifest.json",
            },
            "artifact_count": len(list(exp_dir.rglob("*"))),
        },
    )
    write_json(exp_dir / "artifacts_index.json", {"files": list_artifact_files(exp_dir)})


def maybe_sync_experiment_to_s3(exp_dir: Path) -> str | None:
    """If ``ALPHA_EARTH_EXPERIMENT_SYNC_S3`` is set to ``s3://bucket/prefix/``, run ``aws s3 sync``."""
    uri = os.environ.get("ALPHA_EARTH_EXPERIMENT_SYNC_S3", "").strip()
    if not uri.startswith("s3://"):
        return None
    dest = uri.rstrip("/") + "/" + exp_dir.name + "/"
    try:
        r = subprocess.run(
            ["aws", "s3", "sync", str(exp_dir), dest],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        log = exp_dir / "s3_sync.log"
        log.write_text(
            f"dest={dest}\nreturncode={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}",
            encoding="utf-8",
        )
        return dest if r.returncode == 0 else None
    except Exception as e:
        (exp_dir / "s3_sync_error.txt").write_text(str(e), encoding="utf-8")
        return None


def append_metrics_csv(
    path: Path,
    row: dict[str, Any],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = fieldnames if fieldnames is not None else list(row.keys())
    write_header = not path.is_file()
    out = {k: row.get(k, "") for k in cols}
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(out)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
