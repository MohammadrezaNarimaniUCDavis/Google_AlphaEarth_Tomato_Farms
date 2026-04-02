"""Load modeling YAML + optional CLI overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        out = yaml.safe_load(f)
    if not isinstance(out, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return out


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out
