# Figures

Export paper-quality figures from notebooks (PDF, SVG, or high-DPI PNG) into this folder. The LaTeX under `paper/` can reference copies in `paper/figures/` or `../figures/` depending on your `\\graphicspath` setup.

Suggested naming: `map_study_area.pdf`, `learning_curve.png`, `confusion_matrix.png`.

## From archived SageMaker experiments

After `git pull`, the repo includes `archive/sagemaker-studio/project-outputs/experiments/<run_id>/` with `metrics_epoch.csv`, `confusion_test.json`, and `metrics_test.json`.

Install plotting deps and generate default PNGs into this folder:

```bash
pip install -e ".[figures]"
python tools/plot_experiment_figures.py --out-dir figures
```

Override the experiment directory if you add a new run under `archive/` or copy outputs locally:

```bash
python tools/plot_experiment_figures.py --experiment-dir path/to/experiment --out-dir figures --confusion-split test
```

This writes `learning_curves.png`, `metrics_bar_test.png`, and three confusion figures when `confusion_test.json` (or val) exists:

- `confusion_matrix_<split>.png` — each cell is the **fraction of all pixels** (same information as % of total).
- `confusion_matrix_<split>_row_normalized.png` — **each row sums to 1** (conditional distribution over predicted class given true class).
- `confusion_matrix_<split>_col_normalized.png` — **each column sums to 1** (conditional distribution over true class given predicted class).

## Pixel-level maps (inference)

For probability rasters and masks per chip, use `modeling/inference/infer_chip.py` with `--geotiff` and local data under `data/` (set `ALPHA_EARTH_DATA_SOURCE=auto` if documented in the inference guide). Zonal summaries use `modeling/inference/zonal_stats.py` on the probability GeoTIFF and farm polygons.
