"""SYDE 572 Assignment 2, Part 1(a): polynomial basis regression."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "assn2 material" / "housing.data"
OUTPUT_DIR = HERE / "output"


def polynomial_design_matrix(x_normalized: np.ndarray, degree: int) -> np.ndarray:
    """Return [1, x_1, ..., x_D, x_1^2, ..., x_D^degree].

    The assignment asks for powers of individual variables only, so this function
    deliberately does not generate cross-terms such as x_1*x_2.
    """
    if degree < 1:
        raise ValueError("degree must be at least 1")

    blocks = [np.ones((x_normalized.shape[0], 1))]
    blocks.extend(x_normalized**power for power in range(1, degree + 1))
    return np.hstack(blocks)


def rms_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Compute root-mean-square prediction error."""
    return float(np.sqrt(np.mean((y_true - y_predicted) ** 2)))


def polynomial_regression(
    data_path: Path = DATA_PATH,
    degrees=range(1, 8),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit degrees 1-7 and return degrees, training RMS, and testing RMS.

    The first 100 observations form the training set, exactly as specified. Input
    normalization parameters are learned only from those observations and then
    reused on the test set, preventing information leakage.
    """
    data = np.loadtxt(data_path)
    x, y = data[:, :-1], data[:, -1]
    x_train, x_test = x[:100], x[100:]
    y_train, y_test = y[:100], y[100:]

    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0)
    # CHAS happens to be 0 for every one of the first 100 observations. Giving
    # a constant training column a scale of 1 avoids division by zero; after
    # centering, that entire training column is zero and receives no usable
    # coefficient, which is exactly what the available training data imply.
    train_std = np.where(train_std == 0, 1.0, train_std)

    x_train_normalized = (x_train - train_mean) / train_std
    x_test_normalized = (x_test - train_mean) / train_std

    degree_values = np.asarray(list(degrees), dtype=int)
    training_errors = np.empty(degree_values.size)
    testing_errors = np.empty(degree_values.size)

    for index, degree in enumerate(degree_values):
        phi_train = polynomial_design_matrix(x_train_normalized, int(degree))
        phi_test = polynomial_design_matrix(x_test_normalized, int(degree))

        # lstsq solves the unregularized least-squares problem more stably than
        # explicitly forming and inverting Phi.T @ Phi.
        weights, *_ = np.linalg.lstsq(phi_train, y_train, rcond=None)
        training_errors[index] = rms_error(y_train, phi_train @ weights)
        testing_errors[index] = rms_error(y_test, phi_test @ weights)

    return degree_values, training_errors, testing_errors


def save_error_plot(
    degrees: np.ndarray,
    training_errors: np.ndarray,
    testing_errors: np.ndarray,
    output_path: Path,
) -> None:
    """Save the report figure requested in Part 1(a)."""
    figure, axis = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    axis.plot(degrees, training_errors, "o-", linewidth=2, label="Training RMS")
    axis.plot(degrees, testing_errors, "s-", linewidth=2, label="Testing RMS")
    axis.set(
        title="Polynomial basis regression on Boston Housing",
        xlabel="Polynomial degree",
        ylabel="RMS error (MEDV, $1000s)",
        xticks=degrees,
    )
    # Test error spans many orders of magnitude because higher-order,
    # unregularized polynomials extrapolate far outside the first 100 rows.
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    degrees, training_errors, testing_errors = polynomial_regression()
    output_path = OUTPUT_DIR / "part1a_polynomial_rms.png"
    save_error_plot(degrees, training_errors, testing_errors, output_path)
    results_path = OUTPUT_DIR / "part1a_rms_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        results_path,
        np.column_stack((degrees, training_errors, testing_errors)),
        delimiter=",",
        header="degree,training_rms,testing_rms",
        comments="",
        fmt=["%d", "%.10g", "%.10g"],
    )

    print("Degree  Training RMS  Testing RMS")
    for degree, training_rms, testing_rms in zip(
        degrees, training_errors, testing_errors
    ):
        print(f"{degree:>6d}  {training_rms:>12.6f}  {testing_rms:>11.6f}")
    print(f"\nSaved plot to: {output_path}")
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
