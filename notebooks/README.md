# Notebooks

From the repo root, run **`pip install -e .`** in the same conda env as Jupyter (e.g. `gee`) so `import src...` works without `PYTHONPATH`. See root [`README.md`](../README.md).

| Folder | Contents |
|--------|----------|
| [`landiq/`](landiq/) | LandIQ: [`landiq/years/`](landiq/years/); tomato filter in [`landiq/years/2016/02_filter…`](landiq/years/2016/02_filter_tomato_polygons.ipynb) |
| [`alpha_earth/`](alpha_earth/) | Clip or extract Alpha Earth (multi-year) to tomato polygons |
| [`sagemaker/`](sagemaker/) | Notes for AWS SageMaker + GPU (no code notebook required) |

Suggested order for LandIQ → clips: **landiq** notebooks first, then **alpha_earth**.
