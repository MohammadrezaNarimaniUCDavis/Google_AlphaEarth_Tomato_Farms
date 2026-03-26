"""Survey years and suggested crop-mapping ZIP names (verify against ``data/raw/landiq/<YEAR>/``).

Tomato legend codes **T15** / **T26** match DWR standard land use for recent vintages; always
confirm in that year’s PDF if you use a new edition.
"""

from __future__ import annotations

# Land IQ layers in this project (no 2017 product in your stash).
SURVEY_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)

# Exact zip basename if known; None → use ``find_landiq_crop_zip`` auto-detect in that folder.
ZIP_FILENAME_BY_YEAR: dict[int, str | None] = {
    2018: None,
    2019: None,
    2020: None,
    2021: None,
    2022: None,
    2023: None,
    2024: "i15_crop_mapping_2024_provisional.zip",
}

TOMATO_CODES_DEFAULT: tuple[str, ...] = ("T15", "T26")
CROP_COLUMNS_DEFAULT: tuple[str, ...] = ("CROPTYP1", "CROPTYP2", "CROPTYP3")


def suggested_zip_filename(year: int) -> str | None:
    """Return configured zip name for ``year``, or ``None`` to auto-detect."""
    return ZIP_FILENAME_BY_YEAR.get(int(year))
