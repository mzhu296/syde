"""PCA analysis for SYDE 572 Assignment 3, Part 3.

Preprocessing choice: mean-center each feature, but do not standardize its variance.
This choice must remain unchanged in the later assignment parts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from part2 import COLORS, FEATURE_LABELS, SPECIES, load_iris


OUTPUT_DIR = Path(__file__).with_name("output")


def fit_pca() -> tuple[PCA, np.ndarray, np.ndarray]:
    """Fit four-component PCA to mean-centered, non-standardized Iris data."""
    features, labels = load_iris()
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)

    # Mean-centering only. Feature variances are deliberately not scaled to one.
    X_centered = X - X.mean(axis=0)
    pca = PCA(n_components=4)
    scores = pca.fit_transform(X_centered)
    return pca, scores, y


def save_1d_projection(scores: np.ndarray, y: np.ndarray) -> Path:
    """Plot samples along the first principal component with vertical jitter."""
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    rng = np.random.default_rng(572)
    for species in SPECIES:
        mask = y == species
        jitter = rng.normal(0.0, 0.035, mask.sum())
        ax.scatter(
            scores[mask, 0],
            jitter,
            s=38,
            alpha=0.8,
            color=COLORS[species],
            edgecolors="white",
            linewidths=0.3,
            label=species.removeprefix("Iris-"),
        )
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("PC1 score")
    ax.set_yticks([])
    ax.set_title("Iris data projected onto PC1")
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="x", alpha=0.2)
    output_path = OUTPUT_DIR / "part3_pca_1d.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_2d_projection(scores: np.ndarray, y: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for species in SPECIES:
        mask = y == species
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=42,
            alpha=0.82,
            color=COLORS[species],
            edgecolors="white",
            linewidths=0.35,
            label=species.removeprefix("Iris-"),
        )
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title("Iris PCA projection: PC1 vs PC2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    output_path = OUTPUT_DIR / "part3_pca_2d.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_3d_projection(scores: np.ndarray, y: np.ndarray) -> Path:
    """Plot the top three PC scores from two viewpoints."""
    fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    for panel, (elev, azim) in enumerate(((23, -57), (12, 35)), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        for species in SPECIES:
            mask = y == species
            ax.scatter(
                scores[mask, 0],
                scores[mask, 1],
                scores[mask, 2],
                s=36,
                alpha=0.82,
                color=COLORS[species],
                edgecolors="white",
                linewidths=0.3,
                label=species.removeprefix("Iris-"),
            )
        ax.set_xlabel("PC1 score", labelpad=8)
        ax.set_ylabel("PC2 score", labelpad=8)
        ax.set_zlabel("PC3 score", labelpad=8)
        ax.set_title(f"View {panel} (elev={elev}, azim={azim})")
        ax.view_init(elev=elev, azim=azim)
        if panel == 1:
            ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Iris PCA projection: first three components")
    output_path = OUTPUT_DIR / "part3_pca_3d.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_results(pca: PCA) -> Path:
    output_path = OUTPUT_DIR / "part3_pca_results.txt"
    component_lines = [
        f"PC{index}: {np.array2string(component, precision=6)}"
        for index, component in enumerate(pca.components_, start=1)
    ]
    explained_lines = [
        f"PC{index}: {ratio:.6f}"
        for index, ratio in enumerate(pca.explained_variance_ratio_, start=1)
    ]
    output_path.write_text(
        "Preprocessing: each feature was mean-centered. Features were NOT "
        "standardized to unit variance. This same choice is used in later parts.\n\n"
        "PCA axis vectors (rows of components_):\n"
        + "\n".join(component_lines)
        + "\n\nExplained variance ratios:\n"
        + "\n".join(explained_lines)
        + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pca, scores, labels = fit_pca()
    print("Preprocessing: mean-centered; NOT standardized to unit variance.")
    print("PCA axis vectors (components_):")
    print(np.array2string(pca.components_, precision=6, suppress_small=True))
    for path in (
        save_1d_projection(scores, labels),
        save_2d_projection(scores, labels),
        save_3d_projection(scores, labels),
        write_results(pca),
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
