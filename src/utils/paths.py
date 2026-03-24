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


def resolve_landiq_shapefile_path(cfg: dict[str, Any], root: Path | None = None) -> Path:
    """LandIQ vector input: ``landiq.input_shapefile`` if set, else the single ``*.shp`` under raw (by year/glob)."""
    root = root or REPO_ROOT
    lc = cfg.get("landiq", {})
    explicit = lc.get("input_shapefile")
    if explicit:
        p = resolve_under_root(str(explicit), root)
        if not p.is_file():
            raise FileNotFoundError(
                f"landiq.input_shapefile not found: {p} (configured as {explicit!r})"
            )
        return p

    raw_rel = cfg.get("data", {}).get("raw_landiQ", "data/raw/landiq")
    raw = resolve_under_root(str(raw_rel), root)
    year = lc.get("year")
    search = (raw / str(year)) if year is not None else raw
    if not search.is_dir():
        raise FileNotFoundError(
            f"LandIQ search folder does not exist: {search}. "
            "Set landiq.year, extract under data/raw/landiq/<year>/, or set landiq.input_shapefile."
        )
    glob_pat = lc.get("shapefile_glob", "*.shp")
    recursive = lc.get("shapefile_recursive", True)
    shps = sorted(search.rglob(glob_pat) if recursive else search.glob(glob_pat))
    if not shps:
        raise FileNotFoundError(
            f"No {glob_pat!r} under {search.resolve()} (recursive={recursive}). "
            "Extract ZIPs into raw, or set landiq.input_shapefile to your .shp path."
        )
    if len(shps) != 1:
        raise ValueError(
            f"Expected one shapefile under {search}, got {len(shps)}: {[s.name for s in shps]}. "
            "Tighten landiq.shapefile_glob or set landiq.input_shapefile."
        )
    return shps[0]


def landiq_tomato_gpkg_path(cfg: dict[str, Any], root: Path | None = None) -> Path:
    """Path to the filtered tomato-only GeoPackage (all LandIQ columns preserved for kept rows)."""
    root = root or REPO_ROOT
    sub = cfg.get("data", {}).get("derived_tomato", "data/derived/landiq_tomato")
    base = root / sub if not Path(sub).is_absolute() else Path(sub)
    lc = cfg.get("landiq", {})
    fn = lc.get("output_filename")
    if fn:
        return (base / fn).resolve()
    yr = lc.get("year")
    if yr is not None:
        return (base / f"landiq_tomato_{yr}.gpkg").resolve()
    return (base / "landiq_tomato.gpkg").resolve()
