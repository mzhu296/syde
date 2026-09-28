import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_distance_newton(
    x0,
    y0,
    f,
    df,
    ddf,
    initial_guess=0.0,
    tolerance=1e-7,
    max_iter=100
):
    """
    Newton-Raphson.
    """

    x = initial_guess

    # Store intermediate steps for plotting later
    history = []

    for iteration in range(max_iter):


        D = (x - x0)**2 + (f(x) - y0)**2

        # First derivative of squared distance
        D_prime = (
            2 * (x - x0)
            + 2 * (f(x) - y0) * df(x)
        )

        # Second derivative of squared distance
        D_double_prime = (
            2
            + 2 * df(x)**2
            + 2 * (f(x) - y0) * ddf(x)
        )

        # Save current iteration
        history.append({
            "iteration": iteration,
            "x": x,
            "y": f(x),
            "D": D,
            "D_prime": D_prime,
            "D_double_prime": D_double_prime
        })

        # Avoid division by zero
        if abs(D_double_prime) < 1e-12:
            print("Newton method stopped: second derivative is too close to zero.")
            break

        # Newton-Raphson update
        next_x = x - D_prime / D_double_prime

        # Check convergence
        if abs(next_x - x) < tolerance:
            x = next_x       
            break

        x = next_x

    # Closest point on the function
    closest_x = x
    closest_y = f(x)

    # Actual Euclidean distance
    shortest_distance = math.sqrt(
        (closest_x - x0)**2
        + (closest_y - y0)**2
    )

    return shortest_distance, closest_x, closest_y, history


def plot_newton_steps(points, f, df, ddf, output_path, initial_guess=0.0):
    """Plot the Newton iterates and final shortest-distance segment."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()

    for axis, (x0, y0) in zip(axes, points):
        distance, closest_x, closest_y, history = find_distance_newton(
            x0, y0, f, df, ddf, initial_guess=initial_guess
        )
        iterates = np.array([step["x"] for step in history] + [closest_x])
        curve_x = np.linspace(-10, 10, 800)

        axis.plot(curve_x, f(curve_x), color="#17365d", linewidth=2.2,
                  label=r"$y=x^2+5$")
        axis.scatter([x0], [y0], marker="*", s=150, color="#d1495b",
                     zorder=5, label=f"Given point ({x0}, {y0})")
        axis.annotate(f"P=({x0}, {y0})", (x0, y0), xytext=(8, 10),
                      textcoords="offset points", fontsize=8, fontweight="bold")
        axis.plot(iterates, f(iterates), "o--", color="#f28e2b", linewidth=1.4,
                  markersize=5, label="Newton iterates")
        for number, x_value in enumerate(iterates[:-1]):
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
        axis.set_title(f"Newton-Raphson: P = ({x0}, {y0})")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")

    axes[-1].axis("off")
    figure.suptitle("Newton-Raphson intermediate steps", fontsize=18, fontweight="bold")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

def f(x):
    return x**2 + 5


def df(x):
    return 2*x


def ddf(x):
    return 2

if __name__ == "__main__":
    points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]

    for x0, y0 in points:
        distance, closest_x, closest_y, history = find_distance_newton(
            x0, y0, f, df, ddf, initial_guess=0
        )
        print(f"Given point: ({x0}, {y0})")
        print(f"Closest point: ({closest_x:.5f}, {closest_y:.5f})")
        print(f"Shortest distance: {distance:.5f}")
        print(f"Iterations: {len(history)}")
        print()

    output_path = Path(__file__).resolve().parent / "media" / "newton_steps.png"
    plot_newton_steps(points, f, df, ddf, output_path)
    print(f"Saved Newton-Raphson plot to:\n{output_path}")
