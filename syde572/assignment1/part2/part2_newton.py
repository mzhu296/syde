"""Coordinate-wise Newton-Raphson least-squares fitting for Part Two."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


X_DATA = np.array([0.0, 1.0, 2.0, 3.0])
Y_DATA = np.array([0.5, 1.5, 3.5, 7.5])
OUTPUT_DIR = Path(__file__).resolve().parent / "media"


def coordinate_newton(design_matrix, targets, tolerance=1e-10, max_iter=5000):
    """Minimize SSE by applying one exact Newton update per parameter."""
    coefficients = np.zeros(design_matrix.shape[1], dtype=float)

    def snapshot(iteration):
        residual = design_matrix @ coefficients - targets
        sse = float(residual @ residual)
        return {
            "iteration": iteration,
            "coefficients": coefficients.copy(),
            "sse": sse,
            "mse": sse / len(targets),
        }

    history = [snapshot(0)]
    for iteration in range(1, max_iter + 1):
        previous = coefficients.copy()

        # Gauss-Seidel coordinate updates: later parameters use earlier updates
        # from the same sweep, as requested in the assignment.
        for parameter in range(design_matrix.shape[1]):
            residual = design_matrix @ coefficients - targets
            column = design_matrix[:, parameter]
            gradient = 2.0 * float(column @ residual)
            second_derivative = 2.0 * float(column @ column)
            coefficients[parameter] -= gradient / second_derivative

        history.append(snapshot(iteration))
        if np.max(np.abs(coefficients - previous)) < tolerance:
            break

    return coefficients, history


def write_history(name, history):
    path = OUTPUT_DIR / f"{name}_newton_iterations.csv"
    fieldnames = ["iteration", "c0", "c1", "c2", "sse", "mse"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for step in history:
            values = list(step["coefficients"])
            writer.writerow({
                "iteration": step["iteration"],
                "c0": values[0],
                "c1": values[1] if len(values) > 1 else "",
                "c2": values[2] if len(values) > 2 else "",
                "sse": step["sse"],
                "mse": step["mse"],
            })
    return path


def selected_steps(history):
    requested = [0, 1, 2, 3, 5, 10, 20, 50, len(history) - 1]
    return [history[i] for i in sorted(set(i for i in requested if i < len(history)))]


def plot_fit(name, history, powers, output_path):
    figure, (fit_axis, error_axis) = plt.subplots(
        1, 2, figsize=(14, 5.5), constrained_layout=True
    )
    plot_x = np.linspace(-0.25, 3.25, 500)
    steps = selected_steps(history)
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(steps)))

    fit_axis.scatter(
        X_DATA, Y_DATA, marker="*", s=180, color="#d1495b",
        zorder=8, label="Observed data"
    )
    for color, step in zip(colors, steps):
        coefficients = step["coefficients"]
        curve = sum(c * plot_x**power for c, power in zip(coefficients, powers))
        fit_axis.plot(
            plot_x, curve, color=color, linewidth=2,
            label=f"Iteration {step['iteration']} · MSE {step['mse']:.4f}"
        )
    fit_axis.set_title(f"{name.title()} fit through Newton updates")
    fit_axis.set_xlabel("x")
    fit_axis.set_ylabel("y")
    fit_axis.grid(alpha=0.25)
    fit_axis.legend(fontsize=8, loc="best")

    iterations = [step["iteration"] for step in history]
    mse_values = [step["mse"] for step in history]
    error_axis.semilogy(iterations, mse_values, color="#159a91", linewidth=2.4)
    error_axis.scatter(iterations, mse_values, color="#f28e2b", s=14, zorder=4)
    error_axis.set_title("MSE convergence")
    error_axis.set_xlabel("Full parameter-update sweep")
    error_axis.set_ylabel("Mean squared error (log scale)")
    error_axis.grid(alpha=0.25, which="both")

    figure.suptitle(
        f"Part Two · Coordinate-wise Newton-Raphson · {name.title()}",
        fontsize=17, fontweight="bold"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {
        "line": (np.column_stack([np.ones_like(X_DATA), X_DATA]), [0, 1]),
        "parabola": (
            np.column_stack([np.ones_like(X_DATA), X_DATA, X_DATA**2]),
            [0, 1, 2],
        ),
    }

    for name, (design_matrix, powers) in models.items():
        coefficients, history = coordinate_newton(design_matrix, Y_DATA)
        csv_path = write_history(name, history)
        plot_path = OUTPUT_DIR / f"part2_{name}_newton_steps.png"
        plot_fit(name, history, powers, plot_path)
        print(f"{name.title()}: coefficients={coefficients}")
        print(f"Iterations={len(history) - 1}, MSE={history[-1]['mse']:.8f}")
        print(f"Saved {csv_path}")
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
