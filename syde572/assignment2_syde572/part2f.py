"""SYDE 572 Assignment 2, Part 2(f): regularized Gaussian regression."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "assn2 material" / "housing.data"
OUTPUT_DIR = HERE / "output"
LAMBDAS = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
N_FOLDS = 10
N_BASES = 90
RANDOM_SEED = 572
BANDWIDTH = 1.0


def gaussian_design_matrix(
    x_normalized: np.ndarray,
    centers: np.ndarray,
    bandwidth: float = BANDWIDTH,
) -> np.ndarray:
    """Return a bias column followed by isotropic Gaussian basis functions."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    distances_squared = np.sum(
        (x_normalized[:, None, :] - centers[None, :, :]) ** 2, axis=2
    )
    features = np.exp(-distances_squared / (2.0 * bandwidth**2))
    return np.column_stack((np.ones(x_normalized.shape[0]), features))


def fit_ridge(phi: np.ndarray, y: np.ndarray, regularization: float) -> np.ndarray:
    """Fit L2-regularized regression without penalizing the bias."""
    penalty_root = np.eye(phi.shape[1])
    penalty_root[0, 0] = 0.0
    augmented_phi = np.vstack((phi, np.sqrt(regularization) * penalty_root))
    augmented_y = np.concatenate((y, np.zeros(phi.shape[1])))
    weights, *_ = np.linalg.lstsq(augmented_phi, augmented_y, rcond=None)
    return weights


def rms_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_predicted) ** 2)))


def make_folds(n_observations: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return deterministic, randomly shuffled validation folds."""
    indices = np.random.default_rng(seed).permutation(n_observations)
    return list(np.array_split(indices, n_folds))


def gaussian_regression_reg(
    data_path: Path = DATA_PATH,
    lambdas: np.ndarray = LAMBDAS,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
    bandwidth: float = BANDWIDTH,
) -> dict[str, np.ndarray | float | int]:
    """Select lambda for a 90-basis Gaussian model by 10-fold CV.

    In each fold, the 90 fitting observations are the 90 Gaussian centers. No
    validation observation is used for normalization, center selection, or
    weight fitting.
    """
    data = np.loadtxt(data_path)
    x = data[:100, :-1]
    y = data[:100, -1]
    folds = make_folds(len(x), n_folds, seed)
    fold_errors = np.empty((len(lambdas), n_folds))
    all_indices = np.arange(len(x))

    for fold_number, validation_indices in enumerate(folds):
        fitting_indices = np.setdiff1d(
            all_indices, validation_indices, assume_unique=True
        )
        if len(fitting_indices) != N_BASES:
            raise ValueError(
                f"Expected {N_BASES} fitting observations per fold; "
                f"received {len(fitting_indices)}"
            )

        x_fit, y_fit = x[fitting_indices], y[fitting_indices]
        x_validation, y_validation = x[validation_indices], y[validation_indices]
        mean = x_fit.mean(axis=0)
        std = x_fit.std(axis=0)
        safe_std = np.where(std == 0, 1.0, std)
        z_fit = (x_fit - mean) / safe_std
        z_validation = (x_validation - mean) / safe_std

        # With 90 observations available inside a fold, all 90 are used as
        # centers. This implements the requested basis count without allowing
        # validation observations to leak into the basis construction.
        centers = z_fit
        phi_fit = gaussian_design_matrix(z_fit, centers, bandwidth)
        phi_validation = gaussian_design_matrix(z_validation, centers, bandwidth)

        for lambda_index, regularization in enumerate(lambdas):
            weights = fit_ridge(phi_fit, y_fit, float(regularization))
            fold_errors[lambda_index, fold_number] = rms_error(
                y_validation, phi_validation @ weights
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
        "bandwidth": float(bandwidth),
        "seed": int(seed),
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
        axis.scatter(
            [best_lambda], [best_error], s=110, marker="*", color="#f4a261",
            edgecolor="black", linewidth=0.6, zorder=5,
            label=f"Selected λ={best_lambda:g}",
        )

    axis.set_xscale("log")
    axis.set(
        title="90-basis Gaussian regression: regularization selection",
        xlabel="Regularization constant λ (log scale)",
        ylabel="Average validation RMS (MEDV, $1000s)",
    )
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    result = gaussian_regression_reg()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "part2f_gaussian_regularization_cv.png"
    csv_path = OUTPUT_DIR / "part2f_gaussian_cv_results.csv"
    save_validation_plot(result, plot_path)

    table = np.column_stack((
        result["lambdas"], result["mean_validation_rms"],
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
    print(f"Bandwidth: {result['bandwidth']:g}; random seed: {result['seed']}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved results to: {csv_path}")


if __name__ == "__main__":
    main()
