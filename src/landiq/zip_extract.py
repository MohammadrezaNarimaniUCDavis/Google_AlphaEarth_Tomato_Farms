"""Extract LandIQ zip archives and locate shapefile components."""

from __future__ import annotations

import zipfile
from pathlib import Path


def extract_zip(zip_path: str | Path, out_dir: str | Path, *, clear: bool = False) -> Path:
    """Extract all members of ``zip_path`` into ``out_dir``.

    If ``clear`` is True, remove existing files under ``out_dir`` before extracting
    (only direct children; does not delete ``out_dir`` itself).
    """
    zip_path = Path(zip_path).resolve()
    out_dir = Path(out_dir).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if clear:
        for child in out_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def find_shapefiles(root: str | Path) -> list[Path]:
    """Return sorted paths to ``*.shp`` under ``root`` (recursive)."""
    root = Path(root).resolve()
    return sorted(root.rglob("*.shp"))


def pick_main_shapefile(root: str | Path, *, prefer_name_contains: str | None = "crop") -> Path:
    """Return a single ``.shp`` path under ``root``.

    If multiple exist, prefer a stem/name containing ``prefer_name_contains`` (case-insensitive).
    """
    shps = find_shapefiles(root)
    if not shps:
        raise FileNotFoundError(f"No .shp files under {root}")
    if len(shps) == 1:
        return shps[0]
    if prefer_name_contains:
        key = prefer_name_contains.lower()
        matches = [p for p in shps if key in p.stem.lower()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            shps = matches
    raise ValueError(
        f"Multiple .shp files ({len(shps)}) under {root}; pass an explicit path. Candidates: {shps[:5]}..."
    )


def find_landiq_crop_zip(
    year_folder: str | Path,
    *,
    zip_filename: str | None = None,
) -> Path:
    """Return the crop-mapping ZIP under ``.../landiq/<year>/``.

    If ``zip_filename`` is set, require that file. Otherwise try common glob
    patterns (e.g. ``i15_crop_mapping_*_shp.zip``, ``*crop_mapping*.zip``).
    Raises if none or ambiguous multiple matches.
    """
    year_folder = Path(year_folder).resolve()
    if not year_folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {year_folder}")
    if zip_filename:
        p = year_folder / zip_filename
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    patterns = [
        "i15_crop_mapping_*_shp.zip",
        "*crop_mapping*_shp.zip",
        "*crop_mapping*.zip",
    ]
    uniq: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for p in sorted(year_folder.glob(pat)):
            if p.is_file() and p.suffix.lower() == ".zip" and p not in seen:
                seen.add(p)
                uniq.append(p)
    if not uniq:
        raise FileNotFoundError(
            f"No crop-mapping .zip under {year_folder} (tried patterns: {patterns})"
        )
    if len(uniq) == 1:
        return uniq[0]
    shp_named = [p for p in uniq if "_shp" in p.name.lower()]
    if len(shp_named) == 1:
        return shp_named[0]
    names = [p.name for p in uniq]
    raise ValueError(
        f"Multiple ZIPs in {year_folder}: {names}. Set ZIP_FILENAME in the notebook or config."
    )
