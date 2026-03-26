"""LandIQ 2018 survey metadata and tomato legend codes (DWR standard land use)."""

from .registry import ZIP_FILENAME_BY_YEAR

SURVEY_YEAR = 2018

ZIP_FILENAME = ZIP_FILENAME_BY_YEAR.get(2018)

# DWR legend: T15 processing tomato, T26 market tomato (verify in 2018 legend PDF)
TOMATO_CODES: tuple[str, ...] = ("T15", "T26")

CROP_COLUMNS: tuple[str, ...] = ("CROPTYP1", "CROPTYP2", "CROPTYP3")

LEGEND_PDF_GLOB = "*2018*Legend*.pdf"
