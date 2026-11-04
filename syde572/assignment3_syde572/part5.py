"""Isomap comparison for SYDE 572 Assignment 3, Part 5.

Preprocessing remains consistent with Parts 3 and 4: mean-centering without
standardizing feature variances.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph

from part2 import COLORS, SPECIES, load_iris
from part3 import fit_pca


OUTPUT_DIR = Path(__file__).with_name("output")


def compute_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return PCA and Isomap 2D embeddings plus any Isomap warnings."""
    features, labels = load_iris()
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    X_centered = X - X.mean(axis=0)

    _, pca_scores, _ = fit_pca()
    pca_embedding = pca_scores[:, :2]

    isomap = Isomap(n_neighbors=3, n_components=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        isomap_embedding = isomap.fit_transform(X_centered)
    warning_messages = [str(item.message) for item in caught]
    return pca_embedding, isomap_embedding, y, warning_messages


def classification_accuracy(embedding: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Evaluate a 5-neighbor classifier with stratified 10-fold validation."""
    classifier = KNeighborsClassifier(n_neighbors=5)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=572)
    scores = cross_val_score(classifier, embedding, y, cv=folds, scoring="accuracy")
    return float(scores.mean()), float(scores.std(ddof=1))


NEIGHBOR_SWEEP = (3, 5, 10)


def sweep_isomap(X_centered: np.ndarray, y: np.ndarray) -> dict[int, dict]:
    """Fit Isomap for several k and record embedding, components and accuracy."""
    results: dict[int, dict] = {}
    for k in NEIGHBOR_SWEEP:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding = Isomap(n_neighbors=k, n_components=2).fit_transform(X_centered)
        graph = kneighbors_graph(X_centered, n_neighbors=k, mode="connectivity")
        n_components, _ = connected_components(graph, directed=False)
        accuracy = classification_accuracy(embedding, y)
        results[k] = {
            "embedding": embedding,
            "components": int(n_components),
            "accuracy": accuracy,
        }
    return results


def distance_stress(embedding: np.ndarray, X_centered: np.ndarray) -> float:
    """Relative error between embedded and original pairwise distances."""
    original = pdist(X_centered)
    embedded = pdist(embedding)
    return float(np.sqrt(np.sum((embedded - original) ** 2) / np.sum(original**2)))


def reconstruction_metrics(X_centered: np.ndarray, pca_embedding: np.ndarray) -> dict:
    """Quantify how faithfully each 2D embedding represents the data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isomap = Isomap(n_neighbors=3, n_components=2).fit(X_centered)
    isomap_embedding = isomap.embedding_
    pca_scores_full = fit_pca()[1]
    pca_frobenius = float(
        np.sum(pca_scores_full[:, 2:] ** 2) / np.sum(X_centered**2)
    )
    return {
        "isomap_reconstruction_error": float(isomap.reconstruction_error()),
        "pca_relative_frobenius_error": pca_frobenius,
        "pca_distance_stress": distance_stress(pca_embedding, X_centered),
        "isomap_distance_stress": distance_stress(isomap_embedding, X_centered),
    }


def save_sweep_plot(sweep: dict[int, dict], y: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, len(sweep), figsize=(5 * len(sweep), 4.8), constrained_layout=True)
    for ax, (k, result) in zip(axes, sweep.items()):
        for species in SPECIES:
            mask = y == species
            ax.scatter(
                result["embedding"][mask, 0],
                result["embedding"][mask, 1],
                s=30,
                alpha=0.82,
                color=COLORS[species],
                edgecolors="white",
                linewidths=0.3,
                label=species.removeprefix("Iris-"),
            )
        ax.set_title(f"Isomap k = {k} ({result['components']} graph component(s))")
        ax.set_xlabel("Isomap 1")
        ax.set_ylabel("Isomap 2")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("Effect of the neighbor count on Isomap")
    output_path = OUTPUT_DIR / "part5_isomap_k_sweep.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_comparison_plot(
    pca_embedding: np.ndarray,
    isomap_embedding: np.ndarray,
    y: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    panels = (
        (axes[0], pca_embedding, "PCA: first two components", "PC1", "PC2"),
        (axes[1], isomap_embedding, "Isomap: k = 3", "Isomap 1", "Isomap 2"),
    )
    for ax, embedding, title, x_label, y_label in panels:
        for species in SPECIES:
            mask = y == species
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=40,
                alpha=0.82,
                color=COLORS[species],
                edgecolors="white",
                linewidths=0.35,
                label=species.removeprefix("Iris-"),
            )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("Two-dimensional Iris embeddings")
    output_path = OUTPUT_DIR / "part5_pca_vs_isomap.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_report(
    pca_accuracy: tuple[float, float],
    isomap_accuracy: tuple[float, float],
    warning_messages: list[str],
    sweep: dict[int, dict],
    metrics: dict,
) -> Path:
    warning_text = (
        "Isomap warning(s) observed:\n- " + "\n- ".join(warning_messages)
        if warning_messages
        else "No disconnected-neighbor-graph warning occurred in this run."
    )
    sweep_lines = "\n".join(
        f"k = {k}: {r['components']} connected component(s), 2D kNN accuracy "
        f"{r['accuracy'][0]:.3f} +/- {r['accuracy'][1]:.3f}"
        for k, r in sweep.items()
    )
    report = (
        "Preprocessing: mean-centered; not standardized to unit variance.\n\n"
        "Method comparison\n"
        "PCA is linear, has a closed-form eigendecomposition/SVD solution, supplies "
        "an explained-variance compression measure, and supports inverse "
        "transformation and reconstruction-error analysis. Isomap instead builds a "
        "nearest-neighbor graph and attempts to preserve graph-geodesic distances, "
        "so it can unroll a curved manifold. It does not provide a PCA-style "
        "explained-variance measure or a simple inverse reconstruction.\n\n"
        "Quantitative comparison\n"
        "A 5-neighbor classifier was evaluated on each completed 2D embedding using "
        "stratified 10-fold cross-validation. Because the embeddings were fitted to "
        "the full dataset before validation, these scores are descriptive rather "
        "than a leakage-free estimate of an end-to-end prediction pipeline.\n"
        f"PCA 2D accuracy: {pca_accuracy[0]:.3f} +/- {pca_accuracy[1]:.3f}\n"
        f"Isomap 2D accuracy: {isomap_accuracy[0]:.3f} +/- {isomap_accuracy[1]:.3f}\n\n"
        f"Neighbor-graph diagnostic\n{warning_text}\n\n"
        "The setosa points form their own connected component. The graph stays "
        "disconnected for every k tried (3, 5, 10), because setosa is separated from "
        "the other two species by a gap larger than its k-nearest-neighbor links "
        "can bridge. sklearn joins disconnected components with extra edges before "
        "the geodesic computation, so setosa is compressed into a thin strip in the "
        "embedding and its internal spread is not preserved; increasing k changes "
        "that shape but does not remove the gap.\n\n"
        f"Neighbor sweep\n{sweep_lines}\n\n"
        "Reconstruction / distortion (2D embeddings)\n"
        f"PCA relative Frobenius reconstruction error ||X - X_hat||_F^2/||X||_F^2: "
        f"{metrics['pca_relative_frobenius_error']:.4f}\n"
        f"Isomap (k=3) reconstruction_error() (kernel-space, geodesic distances): "
        f"{metrics['isomap_reconstruction_error']:.4f}\n"
        f"Relative pairwise-distance stress vs original Euclidean distances: "
        f"PCA {metrics['pca_distance_stress']:.4f}, "
        f"Isomap {metrics['isomap_distance_stress']:.4f}\n"
        "Isomap's reconstruction_error() is measured between geodesic-distance "
        "kernels, so it is not on the same scale as PCA's data-space error; the "
        "pairwise-distance stress is the common yardstick.\n\n"
        "Conclusion\n"
        "The Iris structure is already close to linear: both embeddings show a "
        "cleanly separated setosa cluster and partial versicolor/virginica overlap. "
        "Isomap gains little here. PCA is the better compression choice because its "
        "two-dimensional representation retains 97.76% of the variance, is directly "
        "invertible, and has a meaningful reconstruction error.\n"
    )
    output_path = OUTPUT_DIR / "part5_isomap_report.txt"
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pca_embedding, isomap_embedding, labels, warning_messages = compute_embeddings()
    pca_accuracy = classification_accuracy(pca_embedding, labels)
    isomap_accuracy = classification_accuracy(isomap_embedding, labels)
    plot_path = save_comparison_plot(pca_embedding, isomap_embedding, labels)
    features, _ = load_iris()
    X = np.asarray(features, dtype=float)
    sweep = sweep_isomap(X - X.mean(axis=0), labels)
    sweep_path = save_sweep_plot(sweep, labels)
    metrics = reconstruction_metrics(X - X.mean(axis=0), pca_embedding)
    report_path = write_report(pca_accuracy, isomap_accuracy, warning_messages, sweep, metrics)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"PCA 2D kNN accuracy: {pca_accuracy[0]:.3f} +/- {pca_accuracy[1]:.3f}")
    print(f"Isomap 2D kNN accuracy: {isomap_accuracy[0]:.3f} +/- {isomap_accuracy[1]:.3f}")
    if warning_messages:
        print("Isomap warning(s):")
        for message in warning_messages:
            print(f"- {message}")
    else:
        print("No disconnected-neighbor-graph warning occurred.")
    for k, result in sweep.items():
        print(f"k={k}: components={result['components']}, accuracy={result['accuracy'][0]:.3f}")
    print(f"Saved {plot_path}")
    print(f"Saved {sweep_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
