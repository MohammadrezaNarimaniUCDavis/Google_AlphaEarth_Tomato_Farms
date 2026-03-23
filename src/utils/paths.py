"""Load YAML path config from configs/paths.local.yaml with fallback to example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths_config(config_path: Path | None = None) -> dict[str, Any]:
    root = REPO_ROOT
    local = config_path or (root / "configs" / "paths.local.yaml")
    example = root / "configs" / "paths.example.yaml"
    path = local if local.is_file() else example
    if not path.is_file():
        raise FileNotFoundError(f"No config at {local} or {example}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_under_root(relative: str, root: Path | None = None) -> Path:
    root = root or REPO_ROOT
    p = Path(relative)
    return (root / p).resolve() if not p.is_absolute() else p
