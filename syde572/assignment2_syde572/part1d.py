"""SYDE 572 Assignment 2, Part 1(d): regularized polynomial regression."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "assn2 material" / "housing.data"
OUTPUT_DIR = HERE / "output"
LAMBDAS = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
DEGREE = 8
N_FOLDS = 10
RANDOM_SEED = 572


def polynomial_design_matrix_1d(x_normalized: np.ndarray, degree: int) -> np.ndarray:
    """Return [1, z, z^2, ..., z^degree] for a one-dimensional input."""
    return np.vander(np.asarray(x_normalized), N=degree + 1, increasing=True)


def rms_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_predicted) ** 2)))


def fit_ridge(phi: np.ndarray, y: np.ndarray, regularization: float) -> np.ndarray:
    """Fit ridge regression without penalizing the bias coefficient."""
    penalty_root = np.eye(phi.shape[1])
    penalty_root[0, 0] = 0.0
    augmented_phi = np.vstack((phi, np.sqrt(regularization) * penalty_root))
    augmented_y = np.concatenate((y, np.zeros(phi.shape[1])))
    weights, *_ = np.linalg.lstsq(augmented_phi, augmented_y, rcond=None)
    return weights


def make_folds(n_observations: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return deterministic, randomly shuffled validation folds."""
    if n_observations % n_folds != 0:
        raise ValueError("This implementation requires equally sized folds")
    indices = np.random.default_rng(seed).permutation(n_observations)
    return list(np.split(indices, n_folds))


def polynomial_regression_reg(
    data_path: Path = DATA_PATH,
    lambdas: np.ndarray = LAMBDAS,
    degree: int = DEGREE,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> dict[str, np.ndarray | float]:
    """Evaluate degree-8 ridge models with 10-fold cross-validation.

    Only the first 100 observations and the second feature (ZN) are used. For
    each fold, normalization parameters are learned from that fold's 90 fitting
    observations and applied to its 10 validation observations.
    """
    data = np.loadtxt(data_path)
    x = data[:100, 1]
    y = data[:100, -1]
    folds = make_folds(len(x), n_folds, seed)
    fold_errors = np.empty((len(lambdas), n_folds))

    all_indices = np.arange(len(x))
    for fold_number, validation_indices in enumerate(folds):
        fitting_indices = np.setdiff1d(all_indices, validation_indices,
                                       assume_unique=True)
        x_fit, y_fit = x[fitting_indices], y[fitting_indices]
        x_validation, y_validation = x[validation_indices], y[validation_indices]

        mean = x_fit.mean()
        std = x_fit.std()
        z_fit = (x_fit - mean) / std
        z_validation = (x_validation - mean) / std
        phi_fit = polynomial_design_matrix_1d(z_fit, degree)
        phi_validation = polynomial_design_matrix_1d(z_validation, degree)

        for lambda_index, regularization in enumerate(lambdas):
            weights = fit_ridge(phi_fit, y_fit, float(regularization))
            predictions = phi_validation @ weights
            fold_errors[lambda_index, fold_number] = rms_error(
                y_validation, predictions
            )

    mean_validation_rms = fold_errors.mean(axis=1)
    standard_error = fold_errors.std(axis=1, ddof=1) / np.sqrt(n_folds)
    best_index = int(np.argmin(mean_validation_rms))
    return {
        "lambdas": np.asarray(lambdas),
        "fold_errors": fold_errors,
        "mean_validation_rms": mean_validation_rms,
        "standard_error": standard_error,
        "best_lambda": float(lambdas[best_index]),
        "best_validation_rms": float(mean_validation_rms[best_index]),
    }


def save_validation_plot(result: dict, output_path: Path) -> None:
    lambdas = result["lambdas"]
    errors = result["mean_validation_rms"]
    standard_errors = result["standard_error"]
    positive = lambdas > 0

    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    axis.errorbar(
        lambdas[positive], errors[positive], yerr=standard_errors[positive],
        marker="o", linewidth=2.2, capsize=4, color="#277da1",
        label="Mean 10-fold validation RMS ± 1 SE",
    )
    axis.axhline(
        errors[0], color="#d1495b", linestyle="--", linewidth=2,
        label=f"Unregularized baseline (λ=0): {errors[0]:.3f}",
    )
    best_lambda = result["best_lambda"]
    best_error = result["best_validation_rms"]
    if best_lambda > 0:
        axis.scatter([best_lambda], [best_error], s=100, marker="*",
                     color="#f4a261", edgecolor="black", linewidth=0.6,
                     zorder=5, label=f"Selected λ={best_lambda:g}")

    axis.set_xscale("log")
    axis.set(
        title="Degree-8 polynomial regression: regularization selection",
        xlabel="Regularization constant λ (log scale)",
        ylabel="Average validation RMS (MEDV, $1000s)",
    )
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    result = polynomial_regression_reg()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "part1d_regularization_cv.png"
    csv_path = OUTPUT_DIR / "part1d_cv_results.csv"
    save_validation_plot(result, plot_path)

    table = np.column_stack((
        result["lambdas"],
        result["mean_validation_rms"],
        result["standard_error"],
    ))
    np.savetxt(
        csv_path, table, delimiter=",",
        header="lambda,mean_validation_rms,standard_error",
        comments="", fmt="%.10g",
    )

    print("Lambda    Mean validation RMS    Standard error")
    for regularization, error, standard_error in table:
        print(f"{regularization:>7g}    {error:>19.6f}    {standard_error:>14.6f}")
    print(f"\nSelected lambda: {result['best_lambda']:g}")
    print(f"Validation RMS:  {result['best_validation_rms']:.6f}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved results to: {csv_path}")


if __name__ == "__main__":
    main()
