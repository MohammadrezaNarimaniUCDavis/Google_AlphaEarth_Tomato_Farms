"""Resolve chip paths for local disk vs S3 (rasterio /vsis3/)."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def prefer_local_else_s3(local_path: str | Path, s3_uri: str | None) -> str:
    """Return a path string rasterio can open: local file or GDAL /vsis3/ path."""
    lp = Path(local_path)
    if lp.is_file():
        return str(lp.resolve())
    if s3_uri and str(s3_uri).strip().lower().startswith("s3://"):
        return s3_uri_to_vsis3(str(s3_uri).strip())
    raise FileNotFoundError(
        f"Chip not found locally ({lp}) and no usable s3_uri ({s3_uri!r}). "
        "Sync data to S3 and ensure chips_index.csv has s3_uri, or keep files under data/derived/alpha_earth_clips/."
    )


def s3_uri_to_vsis3(s3_uri: str) -> str:
    """``s3://bucket/key`` → ``/vsis3/bucket/key`` for GDAL/rasterio."""
    p = urlparse(s3_uri)
    if p.scheme != "s3" or not p.netloc or not p.path.strip("/"):
        raise ValueError(f"Not a valid s3 URI: {s3_uri!r}")
    key = p.path.lstrip("/")
    return f"/vsis3/{p.netloc}/{key}"


def use_s3_first() -> bool:
    """If true, prefer s3_uri when present (e.g. on SageMaker with data only in S3)."""
    return os.environ.get("ALPHA_EARTH_DATA_SOURCE", "").strip().lower() in ("s3", "auto")


def resolve_raster_path(local_path: str | Path, s3_uri: str | None) -> str:
    """Pick local file, else S3, controlled by ALPHA_EARTH_DATA_SOURCE."""
    mode = os.environ.get("ALPHA_EARTH_DATA_SOURCE", "auto").strip().lower()
    lp = Path(local_path)
    if mode == "local":
        if not lp.is_file():
            raise FileNotFoundError(lp)
        return str(lp.resolve())
    if mode == "s3":
        if not s3_uri:
            raise FileNotFoundError("ALPHA_EARTH_DATA_SOURCE=s3 but row has no s3_uri")
        return s3_uri_to_vsis3(str(s3_uri))
    # auto
    if lp.is_file():
        return str(lp.resolve())
    if s3_uri:
        return s3_uri_to_vsis3(str(s3_uri))
    raise FileNotFoundError(f"No local file and no s3_uri: {lp}")
