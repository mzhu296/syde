"""SYDE 572 Assignment 2, Part 2(e): Gaussian basis regression."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "assn2 material" / "housing.data"
OUTPUT_DIR = HERE / "output"
BASIS_COUNTS = np.arange(5, 100, 10)
RANDOM_SEED = 572
BANDWIDTH = 1.0


def squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return all pairwise squared Euclidean distances to the centers."""
    return np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def gaussian_design_matrix(
    x_normalized: np.ndarray,
    centers: np.ndarray,
    bandwidth: float = BANDWIDTH,
) -> np.ndarray:
    """Return a bias column followed by isotropic Gaussian basis functions."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    distances_squared = squared_distances(x_normalized, centers)
    gaussian_features = np.exp(-distances_squared / (2.0 * bandwidth**2))
    return np.column_stack((np.ones(x_normalized.shape[0]), gaussian_features))


def rms_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_predicted) ** 2)))


def gaussian_regression(
    data_path: Path = DATA_PATH,
    basis_counts: np.ndarray = BASIS_COUNTS,
    seed: int = RANDOM_SEED,
    bandwidth: float = BANDWIDTH,
) -> dict[str, np.ndarray | float | int]:
    """Run the Part 2(e) Gaussian-basis experiment without regularization."""
    data = np.loadtxt(data_path)
    x, y = data[:, :-1], data[:, -1]
    x_train, x_test = x[:100], x[100:]
    y_train, y_test = y[:100], y[100:]

    training_mean = x_train.mean(axis=0)
    training_std = x_train.std(axis=0)
    safe_std = np.where(training_std == 0, 1.0, training_std)
    z_train = (x_train - training_mean) / safe_std
    z_test = (x_test - training_mean) / safe_std

    # A single permutation makes the center sets nested: the K=15 model uses
    # the same first five centers as K=5 plus ten more, as the brief requests.
    permutation = np.random.default_rng(seed).permutation(len(z_train))
    training_errors = np.empty(len(basis_counts))
    testing_errors = np.empty(len(basis_counts))
    ranks = np.empty(len(basis_counts), dtype=int)

    for index, basis_count in enumerate(basis_counts):
        centers = z_train[permutation[:basis_count]]
        phi_train = gaussian_design_matrix(z_train, centers, bandwidth)
        phi_test = gaussian_design_matrix(z_test, centers, bandwidth)
        weights, _, rank, _ = np.linalg.lstsq(phi_train, y_train, rcond=None)
        training_errors[index] = rms_error(y_train, phi_train @ weights)
        testing_errors[index] = rms_error(y_test, phi_test @ weights)
        ranks[index] = rank

    return {
        "basis_counts": np.asarray(basis_counts),
        "training_rms": training_errors,
        "testing_rms": testing_errors,
        "ranks": ranks,
        "center_permutation": permutation,
        "bandwidth": float(bandwidth),
        "seed": int(seed),
    }


def save_error_plot(result: dict, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    axis.plot(
        result["basis_counts"], result["training_rms"], "o-",
        linewidth=2.2, label="Training RMS", color="#277da1",
    )
    axis.plot(
        result["basis_counts"], result["testing_rms"], "s-",
        linewidth=2.2, label="Testing RMS", color="#e76f51",
    )
    axis.set(
        title="Gaussian basis-function regression",
        xlabel="Number of Gaussian basis functions",
        ylabel="RMS error (MEDV, $1000s)",
        xticks=result["basis_counts"],
    )
    axis.grid(alpha=0.3)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    result = gaussian_regression()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "part2e_gaussian_rms.png"
    csv_path = OUTPUT_DIR / "part2e_gaussian_rms_results.csv"
    save_error_plot(result, plot_path)

    table = np.column_stack((
        result["basis_counts"], result["training_rms"],
        result["testing_rms"], result["ranks"],
    ))
    np.savetxt(
        csv_path, table, delimiter=",",
        header="basis_functions,training_rms,testing_rms,design_matrix_rank",
        comments="", fmt=["%d", "%.10g", "%.10g", "%d"],
    )

    print("Bases  Training RMS  Testing RMS  Matrix rank")
    for basis_count, training_rms, testing_rms, rank in table:
        print(
            f"{int(basis_count):>5d}  {training_rms:>12.6f}  "
            f"{testing_rms:>11.6f}  {int(rank):>11d}"
        )
    best_index = int(np.argmin(result["testing_rms"]))
    print(
        f"\nLowest observed test RMS: {result['testing_rms'][best_index]:.6f} "
        f"at {result['basis_counts'][best_index]} basis functions"
    )
    print(f"Bandwidth: {result['bandwidth']:g}; random seed: {result['seed']}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved results to: {csv_path}")


if __name__ == "__main__":
    main()
