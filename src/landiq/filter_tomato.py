"""Filter LandIQ polygons to tomato crops. Configure column and values in paths config."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from ..utils.paths import REPO_ROOT, load_paths_config, resolve_under_root


def filter_tomato(
    gdf: gpd.GeoDataFrame,
    crop_column: str,
    tomato_values: list[str | int | float],
) -> gpd.GeoDataFrame:
    if crop_column not in gdf.columns:
        raise KeyError(f"Column {crop_column!r} not in GeoDataFrame columns: {list(gdf.columns)}")
    mask = gdf[crop_column].isin(tomato_values)
    return gdf.loc[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter LandIQ shapes to tomato polygons.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Input vector path (.shp, .gpkg, .geojson). Overrides config if set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (.gpkg recommended). Overrides config if set.",
    )
    args = parser.parse_args()

    cfg = load_paths_config()
    proj = cfg.get("project_root", ".")
    root = Path(proj).resolve() if proj != "." else REPO_ROOT

    landiq_cfg = cfg.get("landiq", {})
    crop_col = landiq_cfg.get("crop_column")
    tomato_vals = landiq_cfg.get("tomato_values") or []
    if not crop_col or not tomato_vals:
        raise SystemExit(
            "Set landiq.crop_column and landiq.tomato_values in configs/paths.local.yaml "
            "(copy from paths.example.yaml) after exploring your shapefile."
        )

    if args.input:
        in_path = args.input.resolve()
    else:
        raw_dir = resolve_under_root(cfg["data"]["raw_landiQ"], root)
        glob_pat = landiq_cfg.get("shapefile_glob", "*.shp")
        year = landiq_cfg.get("year")
        search_root = (raw_dir / str(year)) if year is not None else raw_dir
        if not search_root.is_dir():
            raise SystemExit(f"LandIQ search folder does not exist: {search_root}")
        recursive = landiq_cfg.get("shapefile_recursive", True)
        shps = sorted(search_root.rglob(glob_pat) if recursive else search_root.glob(glob_pat))
        if len(shps) != 1:
            hint = " Extract ZIPs under data/raw/landiq/<year>/ if you see zero matches."
            raise SystemExit(
                f"Expected exactly one shapefile under {search_root} matching {glob_pat!r} "
                f"(recursive={recursive}); found {len(shps)}."
                f"{hint if len(shps) == 0 else ''} "
                f"Set landiq.year, adjust shapefile_glob, or use --input."
            )
        in_path = shps[0]

    if args.output:
        out_path = args.output.resolve()
    else:
        out_path = (root / cfg["data"]["derived_tomato"] / "landiq_tomato.gpkg").resolve()

    gdf = gpd.read_file(in_path)
    tomato = filter_tomato(gdf, crop_col, tomato_vals)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tomato.to_file(out_path, driver="GPKG")
    print(f"Wrote {len(tomato)} features to {out_path}")


if __name__ == "__main__":
    main()
