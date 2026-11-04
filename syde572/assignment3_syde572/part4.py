"""Variance-retention analysis for SYDE 572 Assignment 3, Part 4.

Uses the Part 3 preprocessing choice: mean-centering without standardization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from part2 import load_iris
from part3 import fit_pca


OUTPUT_DIR = Path(__file__).with_name("output")


def analyze_variance() -> tuple[float, int, np.ndarray, float, float, float]:
    """Return retention results and their Frobenius-norm quantities."""
    pca, scores, _ = fit_pca()
    eigenvalues = pca.explained_variance_
    # The explained-variance ratio is lambda_i / sum_j(lambda_j).
    explained_ratios = eigenvalues / eigenvalues.sum()
    variance_in_2d = float(explained_ratios[:2].sum())
    pcs_for_50_percent = int(
        np.searchsorted(np.cumsum(explained_ratios), 0.50) + 1
    )

    features, _ = load_iris()
    X = np.asarray(features, dtype=float)
    X_centered = X - X.mean(axis=0)
    total_energy = float(np.linalg.norm(X_centered, ord="fro") ** 2)

    # For an orthonormal PCA basis, the projected score energy equals the
    # squared Frobenius norm retained by that low-dimensional representation.
    scores_2d = scores[:, :2]
    retained_energy = float(np.linalg.norm(scores_2d, ord="fro") ** 2)
    X_reconstructed_2d = scores_2d @ pca.components_[:2]
    reconstruction_error = float(
        np.linalg.norm(X_centered - X_reconstructed_2d, ord="fro") ** 2
    )
    return (
        variance_in_2d,
        pcs_for_50_percent,
        explained_ratios,
        retained_energy,
        total_energy,
        reconstruction_error,
    )


def save_cumulative_variance_plot(ratios: np.ndarray) -> Path:
    components = np.arange(1, len(ratios) + 1)
    cumulative = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.bar(
        components,
        100 * ratios,
        color="#56B4E9",
        alpha=0.75,
        label="Individual variance",
    )
    ax.plot(
        components,
        100 * cumulative,
        color="#D55E00",
        marker="o",
        linewidth=2.2,
        label="Cumulative variance",
    )
    ax.axhline(50, color="#666666", linestyle="--", linewidth=1, label="50% target")
    for component, value in zip(components, cumulative):
        ax.annotate(
            f"{100 * value:.1f}%",
            (component, 100 * value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xticks(components)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Iris PCA explained variance (unstandardized features)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    output_path = OUTPUT_DIR / "part4_cumulative_variance.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (
        variance_in_2d,
        pcs_for_50_percent,
        ratios,
        retained_energy,
        total_energy,
        reconstruction_error,
    ) = analyze_variance()
    energy_ratio = retained_energy / total_energy

    report = (
        "Preprocessing: mean-centered; not standardized to unit variance.\n\n"
        "If lambda_i is the eigenvalue associated with PC i, its explained-variance "
        "ratio is lambda_i / sum_j(lambda_j). For centered X, total variance is "
        "proportional to ||X||_F^2. After "
        "projection onto the first k orthonormal principal components, retained "
        "variance is proportional to ||X W_k||_F^2. Therefore the retained "
        "fraction is ||X W_k||_F^2 / ||X||_F^2, equivalently the sum of the "
        "first k explained-variance ratios.\n\n"
        f"Explained-variance ratios: {np.array2string(ratios, precision=6)}\n"
        f"1. First two PCs: {variance_in_2d:.6f} = "
        f"{100 * variance_in_2d:.2f}% of total variance.\n"
        f"   Frobenius check: {retained_energy:.6f} / {total_energy:.6f} "
        f"= {energy_ratio:.6f}.\n"
        f"   Two-PC squared reconstruction error ||X - X_hat||_F^2: "
        f"{reconstruction_error:.6f}.\n"
        f"2. PCs required to reach at least 50%: {pcs_for_50_percent}. "
        f"PC1 alone explains {100 * ratios[0]:.2f}%.\n"
        "\nFor comparison only, standardizing to unit variance would change the "
        "ratios to approximately [0.730, 0.229, 0.037, 0.005], but that is not "
        "the preprocessing used in this assignment analysis.\n"
    )

    output_path = OUTPUT_DIR / "part4_variance_analysis.txt"
    output_path.write_text(report, encoding="utf-8")
    plot_path = save_cumulative_variance_plot(ratios)
    print(report, end="")
    print(f"Saved {output_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
