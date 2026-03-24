"""LandIQ 2016 survey metadata and tomato legend codes (DWR standard land use)."""

from .registry import ZIP_FILENAME_BY_YEAR

SURVEY_YEAR = 2016

ZIP_FILENAME = ZIP_FILENAME_BY_YEAR[2016] or "i15_crop_mapping_2016_shp.zip"

# DWR legend: T15 processing tomato, T26 market tomato
TOMATO_CODES: tuple[str, ...] = ("T15", "T26")

CROP_COLUMNS: tuple[str, ...] = ("CROPTYP1", "CROPTYP2", "CROPTYP3")

LEGEND_PDF_GLOB = "*2016*Legend*.pdf"
