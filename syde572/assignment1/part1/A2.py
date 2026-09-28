import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def golden_section_search(
    x0,
    y0,
    f,
    a,
    b,
    tolerance=1e-7,
    max_iter=1000
):
    """
    GoldenSectionSearch
    """

    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2

    # Approximately 0.381966
    resphi = 2 - phi

    # First two interior points
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)

    # Squared Euclidean distance
    def dist_sq(x):
        return (x - x0)**2 + (f(x) - y0)**2

    # Evaluate first two points
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)

    # Save intermediate steps for plotting
    history = []

    iteration = 0

    while abs(b - a) > tolerance and iteration < max_iter:

        # Store current search interval and candidate points
        history.append({
            "iteration": iteration,
            "a": a,
            "b": b,
            "x1": x1,
            "x2": x2,
            "D_x1": f_x1,
            "D_x2": f_x2
        })

        # If x1 gives a smaller distance,
        # minimum must lie in [a, x2]
        if f_x1 < f_x2:

            b = x2

            x2 = x1
            f_x2 = f_x1

            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)

        # Otherwise minimum lies in [x1, b]
        else:

            a = x1

            x1 = x2
            f_x1 = f_x2

            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)

        iteration += 1

    # Final estimate
    best_x = (a + b) / 2
    best_y = f(best_x)

    shortest_distance = math.sqrt(dist_sq(best_x))

    return shortest_distance, best_x, best_y, history


def plot_golden_steps(points, f, output_path, a=-10, b=10):
    """Plot shrinking golden-section intervals and the final distance."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()

    for axis, (x0, y0) in zip(axes, points):
        distance, closest_x, closest_y, history = golden_section_search(
            x0, y0, f, a, b
        )
        curve_x = np.linspace(a, b, 800)
        shown_steps = history[::max(1, len(history) // 8)]
        candidates = [step["x1"] if step["D_x1"] < step["D_x2"] else step["x2"]
                      for step in shown_steps]

        axis.plot(curve_x, f(curve_x), color="#17365d", linewidth=2.2,
                  label=r"$y=x^2+5$")
        axis.scatter([x0], [y0], marker="*", s=150, color="#d1495b",
                     zorder=5, label=f"Given point ({x0}, {y0})")
        axis.annotate(f"P=({x0}, {y0})", (x0, y0), xytext=(8, 10),
                      textcoords="offset points", fontsize=8, fontweight="bold")
        axis.scatter(candidates, [f(x) for x in candidates], c=range(len(candidates)),
                     cmap="plasma", s=42, zorder=5, label="Interval candidates")
        for number, x_value in enumerate(candidates):
            axis.annotate(f"k={number}", (x_value, f(x_value)), xytext=(5, 5),
                          textcoords="offset points", fontsize=8)
        axis.plot([x0, closest_x], [y0, closest_y], color="#2a9d8f",
                  linewidth=2.2, label=f"Minimum distance = {distance:.4f}")
        axis.scatter([closest_x], [closest_y], s=65, color="#2a9d8f", zorder=6)
        axis.annotate(
            f"Q=({closest_x:.3f}, {closest_y:.3f})",
            (closest_x, closest_y), xytext=(8, 10), textcoords="offset points",
            fontsize=8, fontweight="bold", color="#166f69",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8, "ec": "none"}
        )
        axis.set_xlim(a, b)
        axis.set_title(f"Golden section: P = ({x0}, {y0})")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")

    axes[-1].axis("off")
    figure.suptitle("Golden-section search intermediate steps", fontsize=18,
                    fontweight="bold")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

def f(x):
    return x**2 + 5

if __name__ == "__main__":
    points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]

    for x0, y0 in points:
        distance, closest_x, closest_y, history = golden_section_search(
            x0, y0, f, a=-10, b=10
        )
        print(f"Given point: ({x0}, {y0})")
        print(f"Closest point: ({closest_x:.5f}, {closest_y:.5f})")
        print(f"Shortest distance: {distance:.5f}")
        print(f"Iterations: {len(history)}")
        print()

    output_path = Path(__file__).resolve().parent / "media" / "golden_section_steps.png"
    plot_golden_steps(points, f, output_path)
    print(f"Saved golden-section plot to:\n{output_path}")
