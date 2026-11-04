"""Load and visualize the UCI Iris data for Assignment 3, Part 2."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_FILE = Path(__file__).with_name("data") / "iris.data"
OUTPUT_DIR = Path(__file__).with_name("output")
FEATURE_NAMES = (
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
)
FEATURE_LABELS = (
    "Sepal length (cm)",
    "Sepal width (cm)",
    "Petal length (cm)",
    "Petal width (cm)",
)
SPECIES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
COLORS = {
    "Iris-setosa": "#0072B2",
    "Iris-versicolor": "#E69F00",
    "Iris-virginica": "#009E73",
}


def load_iris(path: Path = DATA_FILE) -> tuple[list[list[float]], list[str]]:
    """Return the feature matrix X and species labels y from the UCI file."""
    features: list[list[float]] = []
    labels: list[str] = []

    with path.open(newline="", encoding="utf-8") as data_file:
        for row in csv.reader(data_file):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(f"Expected 5 columns, found {len(row)}: {row}")
            features.append([float(value) for value in row[:4]])
            labels.append(row[4])

    return features, labels


def save_1d_histograms(X: np.ndarray, y: np.ndarray) -> Path:
    """Plot one outlined class histogram for each feature."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for feature_index, ax in enumerate(axes.flat):
        bins = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), 16)
        for species in SPECIES:
            ax.hist(
                X[y == species, feature_index],
                bins=bins,
                histtype="stepfilled",
                alpha=0.25,
                color=COLORS[species],
            )
            ax.hist(
                X[y == species, feature_index],
                bins=bins,
                histtype="step",
                linewidth=2,
                color=COLORS[species],
                label=species.removeprefix("Iris-"),
            )
        ax.set_title(FEATURE_LABELS[feature_index])
        ax.set_xlabel("Feature value (cm)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("Iris feature distributions by species (1D marginals)", fontsize=16)
    output_path = OUTPUT_DIR / "part2_1d_histograms.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_2d_scatter_grid(X: np.ndarray, y: np.ndarray) -> Path:
    """Plot each distinct feature pair once (lower triangle), colored by species."""
    fig, axes = plt.subplots(4, 4, figsize=(13, 13), constrained_layout=True)
    for row in range(4):
        for column in range(4):
            ax = axes[row, column]
            if column > row:
                ax.set_axis_off()
                continue
            if row == column:
                for species in SPECIES:
                    ax.hist(
                        X[y == species, column],
                        bins=12,
                        histtype="stepfilled",
                        alpha=0.45,
                        color=COLORS[species],
                    )
                ax.set_yticks([])
                ax.set_title("count", fontsize=8, color="#666666", pad=2)
            else:
                for species in SPECIES:
                    mask = y == species
                    ax.scatter(
                        X[mask, column],
                        X[mask, row],
                        s=20,
                        alpha=0.8,
                        color=COLORS[species],
                        edgecolors="none",
                        label=species.removeprefix("Iris-"),
                    )

            if row == 3:
                ax.set_xlabel(FEATURE_LABELS[column], fontsize=9)
            if column == 0 and row > 0:
                ax.set_ylabel(FEATURE_LABELS[row], fontsize=9)
            ax.grid(alpha=0.15)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    axes[0, 3].legend(handles, labels, loc="center", frameon=False, fontsize=13)
    axes[0, 3].set_axis_off()
    fig.suptitle("Iris scatter matrix: every pair of features (2D marginals)", fontsize=16)
    output_path = OUTPUT_DIR / "part2_2d_scatter_grid.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_petal_scatter(X: np.ndarray, y: np.ndarray) -> Path:
    """Single large panel for the most separable feature pair."""
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for species in SPECIES:
        mask = y == species
        ax.scatter(
            X[mask, 2],
            X[mask, 3],
            s=45,
            alpha=0.8,
            color=COLORS[species],
            edgecolors="white",
            linewidths=0.4,
            label=species.removeprefix("Iris-"),
        )
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_title("Petal length vs petal width")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    output_path = OUTPUT_DIR / "part2_petal_scatter.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_3d_scatter(X: np.ndarray, y: np.ndarray) -> Path:
    """Plot petal length, petal width, and sepal length in 3D from two views."""
    fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    for panel, (elev, azim) in enumerate(((24, -58), (14, -125)), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        for species in SPECIES:
            mask = y == species
            ax.scatter(
                X[mask, 2],
                X[mask, 3],
                X[mask, 0],
                s=36,
                alpha=0.85,
                color=COLORS[species],
                edgecolors="white",
                linewidths=0.3,
                label=species.removeprefix("Iris-"),
            )
        ax.set_xlabel("Petal length (cm)", labelpad=8)
        ax.set_ylabel("Petal width (cm)", labelpad=8)
        ax.set_zlabel("Sepal length (cm)", labelpad=8)
        ax.set_title(f"View {panel} (elev={elev}, azim={azim})")
        ax.view_init(elev=elev, azim=azim)
        if panel == 1:
            ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Iris species in three feature dimensions (3D marginal)", fontsize=16)
    output_path = OUTPUT_DIR / "part2_3d_scatter.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_observations(X: np.ndarray, y: np.ndarray) -> Path:
    """Summarize cluster structure with numbers computed from the data."""
    non_setosa = y != "Iris-setosa"
    is_virginica = y[non_setosa] == "Iris-virginica"
    lines = ["Best single-threshold versicolor-vs-virginica accuracy per feature:"]
    for index, label in enumerate(FEATURE_LABELS):
        values = X[non_setosa, index]
        best = max(
            max(((values > t) == is_virginica).mean(), ((values <= t) == is_virginica).mean())
            for t in np.unique(values)
        )
        setosa = X[y == "Iris-setosa", index]
        others = X[non_setosa, index]
        gap = "separable" if setosa.max() < others.min() or setosa.min() > others.max() else "overlapping"
        lines.append(f"  {label}: {best:.1%} (setosa vs rest: {gap})")
    lines.append(
        "\nSetosa separates cleanly from the other species on petal length "
        "(1.0-1.9 cm vs >=3.0 cm) and petal width (0.1-0.6 cm vs >=1.0 cm). "
        "Versicolor and virginica overlap in every single feature; the overlap is "
        "smallest for petal width and petal length, and in the petal length versus "
        "petal width plane. Sepal width alone separates the classes poorly. The 3D "
        "view adds little over the petal plane, so versicolor and virginica still "
        "touch."
    )
    output_path = OUTPUT_DIR / "part2_observations.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def generate_visualizations() -> list[Path]:
    features, labels = load_iris()
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    OUTPUT_DIR.mkdir(exist_ok=True)
    return [
        save_1d_histograms(X, y),
        save_2d_scatter_grid(X, y),
        save_petal_scatter(X, y),
        save_3d_scatter(X, y),
        write_observations(X, y),
    ]


if __name__ == "__main__":
    X, y = load_iris()
    print(f"Loaded {len(X)} samples with {len(FEATURE_NAMES)} features each.")
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print(f"Classes: {', '.join(sorted(set(y)))}")
    for output in generate_visualizations():
        print(f"Saved {output}")
