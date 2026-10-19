"""SYDE 572 Assignment 2, Part 1(c): one-dimensional polynomial plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "assn2 material" / "housing.data"
OUTPUT_DIR = HERE / "output"
SELECTED_DEGREES = (1, 3, 7)


def polynomial_design_matrix_1d(x_normalized: np.ndarray, degree: int) -> np.ndarray:
    """Return [1, z, z^2, ..., z^degree] for a one-dimensional input."""
    if degree < 1:
        raise ValueError("degree must be at least 1")
    return np.vander(np.asarray(x_normalized), N=degree + 1, increasing=True)


def rms_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_predicted) ** 2)))


def polynomial_regression_1d_vis(
    degree: int,
    data_path: Path = DATA_PATH,
) -> dict[str, np.ndarray | float | int]:
    """Fit and return everything needed to visualize a 1D polynomial model.

    The second input feature, ZN, is normalized using only the first 100
    observations. The returned curve is evaluated over the full observed ZN
    range but all fitting uses the training observations only.
    """
    data = np.loadtxt(data_path)
    x = data[:, 1]  # Second feature: ZN.
    y = data[:, -1]
    x_train, x_test = x[:100], x[100:]
    y_train, y_test = y[:100], y[100:]

    training_mean = float(x_train.mean())
    training_std = float(x_train.std())
    if training_std == 0:
        raise ValueError("The training ZN feature has zero variance")

    z_train = (x_train - training_mean) / training_std
    z_test = (x_test - training_mean) / training_std
    phi_train = polynomial_design_matrix_1d(z_train, degree)
    phi_test = polynomial_design_matrix_1d(z_test, degree)
    weights, *_ = np.linalg.lstsq(phi_train, y_train, rcond=None)

    curve_x = np.linspace(x.min(), x.max(), 800)
    curve_z = (curve_x - training_mean) / training_std
    curve_y = polynomial_design_matrix_1d(curve_z, degree) @ weights

    return {
        "degree": degree,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "curve_x": curve_x,
        "curve_y": curve_y,
        "weights": weights,
        "training_rms": rms_error(y_train, phi_train @ weights),
        "testing_rms": rms_error(y_test, phi_test @ weights),
        "training_mean": training_mean,
        "training_std": training_std,
    }


def draw_model(axis: plt.Axes, result: dict, show_legend: bool = True) -> None:
    """Draw one fitted model on an existing axis."""
    axis.scatter(
        result["x_test"], result["y_test"], s=25, alpha=0.32,
        color="#e76f51", edgecolors="none", label="Test data", zorder=1,
    )
    axis.scatter(
        result["x_train"], result["y_train"], s=31, alpha=0.68,
        color="#277da1", edgecolors="white", linewidths=0.3,
        label="Training data", zorder=2,
    )
    axis.plot(
        result["curve_x"], result["curve_y"], color="#1b4332",
        linewidth=2.4, label="Learned polynomial", zorder=3,
    )
    axis.set_title(
        f"Degree {result['degree']}\n"
        f"train RMS = {result['training_rms']:.3f}, "
        f"test RMS = {result['testing_rms']:.3f}",
        fontweight="bold",
    )
    axis.set_xlabel("ZN: residential land zoned for lots over 25,000 ft² (%)")
    axis.set_ylabel("MEDV ($1000s)")
    axis.grid(alpha=0.25)
    if show_legend:
        axis.legend(loc="best", fontsize=9)


def save_individual_plot(result: dict, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
    draw_model(axis, result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_combined_plot(results: list[dict], output_path: Path) -> None:
    figure, axes = plt.subplots(
        1, len(results), figsize=(17, 5.3), sharex=True, constrained_layout=True
    )
    for axis, result in zip(axes, results):
        draw_model(axis, result, show_legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3,
                  frameon=True)
    figure.suptitle("One-dimensional polynomial regression using ZN", fontsize=17,
                    fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_overlay_plot(results: list[dict], output_path: Path) -> None:
    """Overlay all selected polynomial curves on one set of axes."""
    reference = results[0]
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.scatter(
        reference["x_test"], reference["y_test"], s=25, alpha=0.28,
        color="#e76f51", edgecolors="none", label="Test data", zorder=1,
    )
    axis.scatter(
        reference["x_train"], reference["y_train"], s=31, alpha=0.62,
        color="#277da1", edgecolors="white", linewidths=0.3,
        label="Training data", zorder=2,
    )
    curve_styles = {
        1: ("#1b4332", "-"),
        3: ("#f4a261", "--"),
        7: ("#6a4c93", "-."),
    }
    for result in results:
        color, linestyle = curve_styles.get(result["degree"], (None, "-"))
        axis.plot(
            result["curve_x"], result["curve_y"], color=color,
            linestyle=linestyle, linewidth=2.5,
            label=(f"Degree {result['degree']} "
                   f"(test RMS {result['testing_rms']:.3f})"),
            zorder=3,
        )

    axis.set(
        title="Overlay of polynomial models using ZN",
        xlabel="ZN: residential land zoned for lots over 25,000 ft² (%)",
        ylabel="MEDV ($1000s)",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    results = [polynomial_regression_1d_vis(degree)
               for degree in SELECTED_DEGREES]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for result in results:
        save_individual_plot(
            result,
            OUTPUT_DIR / f"part1c_zn_degree_{result['degree']}.png",
        )
    combined_path = OUTPUT_DIR / "part1c_zn_polynomial_comparison.png"
    save_combined_plot(results, combined_path)
    overlay_path = OUTPUT_DIR / "part1c_zn_polynomial_overlay.png"
    save_overlay_plot(results, overlay_path)

    csv_path = OUTPUT_DIR / "part1c_rms_results.csv"
    np.savetxt(
        csv_path,
        np.array([
            [result["degree"], result["training_rms"], result["testing_rms"]]
            for result in results
        ]),
        delimiter=",",
        header="degree,training_rms,testing_rms",
        comments="",
        fmt=["%d", "%.10g", "%.10g"],
    )

    print("Degree  Training RMS  Testing RMS")
    for result in results:
        print(
            f"{result['degree']:>6d}  {result['training_rms']:>12.6f}  "
            f"{result['testing_rms']:>11.6f}"
        )
    print(f"\nSaved combined plot to: {combined_path}")
    print(f"Saved overlay plot to: {overlay_path}")
    print(f"Saved results to: {csv_path}")


if __name__ == "__main__":
    main()
