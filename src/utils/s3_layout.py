"""Build S3 URIs and key prefixes from configs/paths*.yaml ``s3`` section."""

from __future__ import annotations

from typing import Any


def _trim_slashes(s: str) -> str:
    return s.strip().strip("/")


def s3_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("s3") or {}


def s3_bucket(cfg: dict[str, Any]) -> str | None:
    """Bucket name or None if unset."""
    b = s3_cfg(cfg).get("bucket")
    return str(b).strip() if b else None


def s3_project_root_prefix(cfg: dict[str, Any]) -> str:
    """Top-level key prefix for this repo inside the bucket (no leading/trailing slashes)."""
    p = s3_cfg(cfg).get("project_root_prefix", "google-alphaearth-tomato-farms")
    return _trim_slashes(str(p)) if p else "google-alphaearth-tomato-farms"


def s3_join_key(cfg: dict[str, Any], *relative_parts: str) -> str:
    """S3 object key under this project: ``<project_root_prefix>/<part>/...`` (bucket not included)."""
    root = s3_project_root_prefix(cfg)
    segs = [root] + [_trim_slashes(str(p)) for p in relative_parts if p and _trim_slashes(str(p))]
    return "/".join(segs)


def s3_uri(cfg: dict[str, Any], *relative_parts: str) -> str | None:
    """``s3://bucket/key/...`` or None if bucket missing in config."""
    b = s3_bucket(cfg)
    if not b:
        return None
    return f"s3://{b}/{s3_join_key(cfg, *relative_parts)}"


def s3_layout_keys(cfg: dict[str, Any]) -> dict[str, str]:
    """Named relative categories under ``project_root_prefix`` (see ``paths.example.yaml``)."""
    keys = s3_cfg(cfg).get("keys") or {}
    defaults = {
        "raw_landiq": "raw/landiq",
        "derived_tomato": "derived/landiq_tomato",
        "derived_non_tomato": "derived/landiq_non_tomato",
        "alpha_earth_clips": "derived/alpha_earth_clips",
        "manifests": "manifests",
        "models": "models",
        "splits": "splits",
    }
    out = {**defaults, **{k: str(v).strip().strip("/") for k, v in keys.items() if v}}
    return out


def s3_category_prefix_uri(cfg: dict[str, Any], category: str) -> str | None:
    """URI to a “folder” prefix for a layout key, e.g. ``alpha_earth_clips`` → ``s3://…/derived/alpha_earth_clips/``."""
    lk = s3_layout_keys(cfg)
    rel = lk.get(category)
    if not rel:
        return None
    u = s3_uri(cfg, rel)
    return u + "/" if u else None
