"""SYDE 572 Assignment 2, Part 1(b): inspect degree-1 weights."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "assn2 material" / "housing.data"
OUTPUT_DIR = HERE / "output"
FEATURE_NAMES = (
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
)


def fit_degree_one(
    data_path: Path = DATA_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit degree-1 regression on the first 100 normalized observations.

    Returns the complete weight vector (bias first), followed by the training
    means and standard deviations used to normalize the 13 input features.
    """
    data = np.loadtxt(data_path)
    x_train = data[:100, :-1]
    y_train = data[:100, -1]

    means = x_train.mean(axis=0)
    standard_deviations = x_train.std(axis=0)

    # CHAS is zero throughout the mandated first-100 training split. Assigning
    # a scale of 1 prevents division by zero; its centered column remains zero,
    # correctly signalling that its coefficient cannot be learned here.
    safe_standard_deviations = np.where(standard_deviations == 0, 1.0,
                                        standard_deviations)
    x_normalized = (x_train - means) / safe_standard_deviations
    design_matrix = np.column_stack((np.ones(x_train.shape[0]), x_normalized))

    weights, *_ = np.linalg.lstsq(design_matrix, y_train, rcond=None)
    return weights, means, standard_deviations


def write_weight_table(
    output_path: Path,
    weights: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> None:
    """Write a labeled CSV containing the bias and every feature weight."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        output.write("weight_index,term,weight,training_mean,training_std\n")
        output.write(f"0,bias,{weights[0]:.10g},,\n")
        for index, (name, weight, mean, std) in enumerate(
            zip(FEATURE_NAMES, weights[1:], means, standard_deviations), start=1
        ):
            output.write(f"{index},{name},{weight:.10g},{mean:.10g},{std:.10g}\n")


def save_weight_plot(output_path: Path, weights: np.ndarray) -> None:
    """Plot the bias separately and the standardized feature weights."""
    feature_weights = weights[1:]
    colors = ["#2a9d8f" if weight >= 0 else "#d1495b"
              for weight in feature_weights]

    figure = plt.figure(figsize=(10, 7.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(1, 5))
    bias_axis = figure.add_subplot(grid[0])
    weight_axis = figure.add_subplot(grid[1])

    bias_axis.barh(["Bias"], [weights[0]], color="#264653", height=0.55)
    bias_axis.text(weights[0], 0, f"  {weights[0]:.3f}", va="center",
                   fontweight="bold")
    bias_axis.set_xlim(0, weights[0] * 1.17)
    bias_axis.set_title("Degree-1 regression weights (normalized inputs)",
                        fontsize=15, fontweight="bold")
    bias_axis.set_xlabel("Weight (MEDV, $1000s)")
    bias_axis.grid(axis="x", alpha=0.25)

    positions = np.arange(len(FEATURE_NAMES))
    weight_axis.barh(positions, feature_weights, color=colors, height=0.7)
    weight_axis.set_yticks(positions, FEATURE_NAMES)
    weight_axis.invert_yaxis()
    weight_axis.axvline(0, color="black", linewidth=0.8)
    weight_axis.set_xlabel(
        "Weight per one training-standard-deviation increase (MEDV, $1000s)"
    )
    weight_axis.grid(axis="x", alpha=0.25)

    offset = max(abs(feature_weights)) * 0.025
    for position, weight in zip(positions, feature_weights):
        horizontal_alignment = "left" if weight >= 0 else "right"
        text_x = weight + offset if weight >= 0 else weight - offset
        weight_axis.text(text_x, position, f"{weight:.3f}", va="center",
                         ha=horizontal_alignment, fontsize=9)
    limit = max(abs(feature_weights)) * 1.2
    weight_axis.set_xlim(-limit, limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    weights, means, standard_deviations = fit_degree_one()
    table_path = OUTPUT_DIR / "part1b_degree1_weights.csv"
    plot_path = OUTPUT_DIR / "part1b_degree1_weights.png"
    write_weight_table(table_path, weights, means, standard_deviations)
    save_weight_plot(plot_path, weights)

    print("Index  Term       Weight")
    print(f"{0:>5d}  {'bias':<7} {weights[0]:>12.6f}")
    for index, (name, weight) in enumerate(
        zip(FEATURE_NAMES, weights[1:]), start=1
    ):
        print(f"{index:>5d}  {name:<7} {weight:>12.6f}")

    # RM and DIS are original features 6 and 8; the bias shifts their positions
    # in the full weight vector to indices 6 and 8, respectively.
    print(f"\nRM weight  (w5 in feature-only notation): {weights[6]:.6f}")
    print(f"DIS weight (8th original feature):        {weights[8]:.6f}")
    print(f"Saved weight table to: {table_path}")
    print(f"Saved weight plot to: {plot_path}")


if __name__ == "__main__":
    main()
