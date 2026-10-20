# SYDE 572 Assignment 1 — Code

Distance minimization (Part I) and least-squares function fitting (Part II).
Report: https://mzhu296.github.io/syde/syde572/assignment1/ieee.html

## Setup

Requires Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Each script writes its figures, animations and CSV files to a `media/` folder next to it.

```bash
cd part1
python A1.py                        # Newton–Raphson distance to y = x^2 + 5 (Fig. 1)
python A2.py                        # Golden-section search distance to y = x^2 + 5 (Fig. 2)
python non_polynomial_examples.py   # Exponential, logarithmic, rational, radical functions (Figs. 5–12)
python animate_distance.py          # Animated searches (Figs. 3–4), ~30 s

cd ../part2
python part2_newton.py              # Coordinate-wise Newton–Raphson line and parabola fits (Figs. 13–14)
python animate_fit.py               # Animated coordinate updates (Figs. 15–16), ~1 min
```

## Files

| File | Contents |
|---|---|
| `part1/A1.py` | `find_distance_newton`: Newton–Raphson on D'(x) = 0 for any f, f', f'' |
| `part1/A2.py` | `golden_section_search`: derivative-free minimization of D(x) on a bracket |
| `part1/non_polynomial_examples.py` | Both methods applied to non-polynomial functions |
| `part1/animate_distance.py` | GIF animations of both searches |
| `part2/part2_newton.py` | `coordinate_newton`: one-coefficient-at-a-time Newton updates on the MSE |
| `part2/animate_fit.py` | GIF animations of the coordinate updates on the MSE contours |

Expected results: line `y = 2.3x - 0.2` (MSE 0.575), parabola `y = 0.75x^2 + 0.05x + 0.55` (MSE 0.0125).
