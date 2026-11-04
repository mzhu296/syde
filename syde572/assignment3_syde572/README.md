# SYDE 572 Assignment 3 — Code

PCA and Isomap on the Iris dataset (Parts 2–5; Part 1 is derived by hand in the report).
Report: https://mzhu296.github.io/syde/syde572/assignment3_syde572/ieee.html

## Setup

Requires Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Each script writes its figures and text summaries to `output/`. Parts 3–5 import helpers from earlier parts, so run them from this folder.

```bash
python part2.py    # 1D/2D/3D marginals of Iris (Figs. 2–5)
python part3.py    # PCA projections and axis vectors (Figs. 6–8)
python part4.py    # explained variance and compression (Fig. 9)
python part5.py    # Isomap (k = 3, 5, 10) vs PCA (Figs. 10–11)
```

## Files

| File | Contents |
|---|---|
| `data/iris.data` | UCI Iris data (150 rows, 4 features, 3 classes) |
| `part2.py` | Loader and the 1D, 2D and 3D marginal plots |
| `part3.py` | `fit_pca`: PCA on mean-centered data, 1D/2D/3D projections |
| `part4.py` | Explained variance, Frobenius-norm check, cumulative-variance plot |
| `part5.py` | Isomap vs PCA, neighbor sweep, 5-NN cross-validated accuracy |
| `ieee.html` | The assignment report |

Features are mean-centered but not standardized in every part.
