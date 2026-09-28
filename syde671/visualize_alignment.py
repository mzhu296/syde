"""Generate alignment-search visualizations for one Prokudin-Gorskii plate.

The default input is data/01007a.jpg. Brute-force heatmaps use the coarsest
pyramid level as a low-resolution version of the same plate; pyramid figures
use the full-resolution input.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import (
    build_pyramid,
    load_image,
    ncc,
    l2_distance,
    search_alignment,
    shift_image,
    split_channels,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def score_surface(moving, reference, metric, search_range=15, border=20):
    """Evaluate every integer displacement in a square search window."""
    shifts = np.arange(-search_range, search_range + 1)
    scores = np.empty((len(shifts), len(shifts)), dtype=np.float64)
    margin = border + search_range
    reference_crop = reference[margin:-margin, margin:-margin]

    if min(reference_crop.shape) <= 0:
        raise ValueError("Low-resolution image is too small for the search window.")

    for row, dy in enumerate(shifts):
        for column, dx in enumerate(shifts):
            shifted = shift_image(moving, int(dx), int(dy))
            shifted_crop = shifted[margin:-margin, margin:-margin]
            if metric == "l2":
                scores[row, column] = l2_distance(shifted_crop, reference_crop)
            elif metric == "ncc":
                scores[row, column] = ncc(shifted_crop, reference_crop)
            else:
                raise ValueError("metric must be 'l2' or 'ncc'")

    if metric == "l2":
        best_row, best_column = np.unravel_index(np.argmin(scores), scores.shape)
    else:
        best_row, best_column = np.unravel_index(np.argmax(scores), scores.shape)

    best = (int(shifts[best_column]), int(shifts[best_row]))
    return shifts, scores, best


def plot_heatmaps(B, G, R, metric, output_path, source_name):
    """Plot green-to-blue and red-to-blue brute-force score surfaces."""
    surfaces = []
    for channel_name, channel in (("Green → Blue", G), ("Red → Blue", R)):
        shifts, scores, best = score_surface(channel, B, metric)
        surfaces.append((channel_name, shifts, scores, best))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    cmap = "viridis_r" if metric == "l2" else "viridis"
    optimum = "minimum" if metric == "l2" else "maximum"

    for axis, (channel_name, shifts, scores, best) in zip(axes, surfaces):
        image = axis.imshow(
            scores,
            origin="lower",
            extent=[shifts[0] - 0.5, shifts[-1] + 0.5,
                    shifts[0] - 0.5, shifts[-1] + 0.5],
            cmap=cmap,
            aspect="equal",
        )
        axis.scatter(*best, marker="x", s=110, linewidths=2.5, color="#d62728")
        axis.annotate(
            f"{optimum}: ({best[0]}, {best[1]})",
            best,
            xytext=(8, 10),
            textcoords="offset points",
            color="#7f0000",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
        )
        axis.set_title(channel_name)
        axis.set_xlabel("Horizontal displacement $d_x$ (pixels)")
        axis.set_ylabel("Vertical displacement $d_y$ (pixels)")
        figure.colorbar(image, ax=axis, shrink=0.82, label=metric.upper() + " score")

    figure.suptitle(
        f"{source_name}: brute-force {metric.upper()} search at the coarsest pyramid level",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def align_with_history(moving, reference, channel_name, metric="ncc"):
    """Run coarse-to-fine alignment and retain every per-level estimate."""
    moving_pyramid = build_pyramid(moving, min_size=150)
    reference_pyramid = build_pyramid(reference, min_size=150)
    level_count = min(len(moving_pyramid), len(reference_pyramid))
    dx = dy = 0
    history = []

    for level in reversed(range(level_count)):
        if level == level_count - 1:
            predicted_dx = predicted_dy = 0
            search_range = 15
        else:
            predicted_dx = dx * 2
            predicted_dy = dy * 2
            dx, dy = predicted_dx, predicted_dy
            search_range = 2

        dx, dy, score = search_alignment(
            moving_pyramid[level],
            reference_pyramid[level],
            center_dx=dx,
            center_dy=dy,
            search_range=search_range,
            metric=metric,
        )
        scale = 2 ** level
        history.append({
            "channel": channel_name,
            "level": level,
            "height": reference_pyramid[level].shape[0],
            "width": reference_pyramid[level].shape[1],
            "predicted_dx": predicted_dx,
            "predicted_dy": predicted_dy,
            "selected_dx": dx,
            "selected_dy": dy,
            "full_resolution_dx": dx * scale,
            "full_resolution_dy": dy * scale,
            "score": float(score),
        })

    return dx, dy, history, reference_pyramid, moving_pyramid


def plot_pyramid_diagram(B, G, R, output_path, source_name):
    """Show the three channels at every pyramid level."""
    pyramids = [build_pyramid(channel, min_size=150) for channel in (B, G, R)]
    count = min(map(len, pyramids))
    figure, axes = plt.subplots(3, count, figsize=(3.0 * count, 7.4), constrained_layout=True)
    row_labels = ("Blue reference", "Green moving", "Red moving")

    if count == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for row, (label, pyramid) in enumerate(zip(row_labels, pyramids)):
        for level in range(count):
            axis = axes[row, level]
            image = pyramid[level]
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.axis("off")
            if row == 0:
                axis.set_title(f"Level {level}\n{image.shape[1]} × {image.shape[0]}")
            if level == 0:
                axis.set_ylabel(label, fontsize=10)
                axis.yaxis.set_label_position("left")

    figure.suptitle(
        f"{source_name}: Gaussian pyramid from full resolution to coarse scale",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trajectory(histories, output_path, source_name):
    """Plot per-level estimates projected into full-resolution coordinates."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
    colors = {"Green": "#2a9d8f", "Red": "#d1495b"}

    for axis, history in zip(axes, histories):
        channel = history[0]["channel"]
        xs = [step["full_resolution_dx"] for step in history]
        ys = [step["full_resolution_dy"] for step in history]
        axis.plot(xs, ys, "o-", linewidth=2.2, markersize=7, color=colors[channel])
        for step, x, y in zip(history, xs, ys):
            axis.annotate(
                f"L{step['level']}\n({step['selected_dx']}, {step['selected_dy']})",
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
            )
        axis.scatter([xs[-1]], [ys[-1]], s=130, facecolors="none",
                     edgecolors="#111111", linewidths=1.8, zorder=5)
        axis.set_title(f"{channel} → Blue")
        axis.set_xlabel("Estimated full-resolution $d_x$ (pixels)")
        axis.set_ylabel("Estimated full-resolution $d_y$ (pixels)")
        axis.grid(alpha=0.25)

    figure.suptitle(
        f"{source_name}: coarse-to-fine NCC displacement trajectory",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def save_trajectory_table(histories, output_path):
    fields = [
        "channel", "level", "width", "height", "predicted_dx", "predicted_dy",
        "selected_dx", "selected_dy", "full_resolution_dx",
        "full_resolution_dy", "score",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for history in histories:
            writer.writerows(history)


def crop_for_display(image, offsets, base_border=20):
    largest = max(abs(value) for value in offsets)
    border = min(base_border + largest, image.shape[0] // 4, image.shape[1] // 4)
    if border:
        return image[border:-border, border:-border]
    return image


def plot_final_comparison(B, G, R, green_offset, red_offset, output_path, source_name):
    unaligned = np.dstack([R, G, B])
    aligned = np.dstack([
        shift_image(R, *red_offset),
        shift_image(G, *green_offset),
        B,
    ])
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    axes[0].imshow(np.clip(crop_for_display(unaligned, (0, 0, 0, 0)), 0, 1))
    axes[0].set_title("Unaligned channel stack")
    axes[1].imshow(np.clip(crop_for_display(
        aligned,
        (*green_offset, *red_offset),
    ), 0, 1))
    axes[1].set_title(
        f"Pyramid NCC aligned\nG {green_offset}, R {red_offset}"
    )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"{source_name}: full-resolution reconstruction",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR / "data" / "01007a.jpg",
        help="Vertically stacked glass-plate image (default: data/01007a.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "visualizations" / "01007a",
        help="Directory for figures and the trajectory CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    B, G, R = split_channels(image)
    source_name = image_path.stem

    B_pyramid = build_pyramid(B, min_size=150)
    G_pyramid = build_pyramid(G, min_size=150)
    R_pyramid = build_pyramid(R, min_size=150)
    B_low, G_low, R_low = B_pyramid[-1], G_pyramid[-1], R_pyramid[-1]

    plot_heatmaps(B_low, G_low, R_low, "l2", output_dir / f"{source_name}_l2_heatmap.png", source_name)
    plot_heatmaps(B_low, G_low, R_low, "ncc", output_dir / f"{source_name}_ncc_heatmap.png", source_name)
    plot_pyramid_diagram(B, G, R, output_dir / f"{source_name}_pyramid_levels.png", source_name)

    g_dx, g_dy, green_history, _, _ = align_with_history(G, B, "Green")
    r_dx, r_dy, red_history, _, _ = align_with_history(R, B, "Red")
    histories = [green_history, red_history]
    plot_trajectory(histories, output_dir / f"{source_name}_pyramid_trajectory.png", source_name)
    save_trajectory_table(histories, output_dir / f"{source_name}_pyramid_trajectory.csv")
    plot_final_comparison(
        B, G, R,
        (g_dx, g_dy),
        (r_dx, r_dy),
        output_dir / f"{source_name}_alignment_comparison.png",
        source_name,
    )

    print(f"Input: {image_path}")
    print(f"Low-resolution heatmap level: {B_low.shape[1]} x {B_low.shape[0]}")
    print(f"Final green offset: ({g_dx}, {g_dy})")
    print(f"Final red offset: ({r_dx}, {r_dy})")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
