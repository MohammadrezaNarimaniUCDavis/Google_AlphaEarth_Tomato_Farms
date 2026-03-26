# Notebooks

From the repo root, run **`pip install -e .`** in the same conda env as Jupyter (e.g. `gee`) so `import src...` works without `PYTHONPATH`. See root [`README.md`](../README.md).

| Folder | Contents |
|--------|----------|
| [`landiq/`](landiq/) | LandIQ: [`landiq/years/`](landiq/years/); filter in [`landiq/years/2018/02_filter…`](landiq/years/2018/02_filter_tomato_polygons.ipynb) |
| [`alpha_earth/`](alpha_earth/) | EE: [`alpha_earth/earth_engine/years/2018/`](alpha_earth/earth_engine/years/2018/); local clip: [`alpha_earth/years/2018/`](alpha_earth/years/2018/) |
| [`sagemaker/`](sagemaker/) | Notes for AWS SageMaker + GPU (no code notebook required) |

Suggested order for LandIQ → clips: **landiq** notebooks first, then **alpha_earth**.
