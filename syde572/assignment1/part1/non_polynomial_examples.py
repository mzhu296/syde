"""Run and visualize both numerical methods on non-polynomial functions."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from A1 import find_distance_newton
from A2 import golden_section_search


POINTS = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
OUTPUT_DIR = Path(__file__).resolve().parent / "media"


EXAMPLES = {
    "exponential": {
        "label": r"$y=e^{x/4}+1$",
        "f": lambda x: np.exp(x / 4) + 1,
        "df": lambda x: np.exp(x / 4) / 4,
        "ddf": lambda x: np.exp(x / 4) / 16,
        "interval": (-10.0, 10.0),
        "initial": 0.0,
    },
    "logarithmic": {
        "label": r"$y=\ln(x+11)+1$",
        "f": lambda x: np.log(x + 11) + 1,
        "df": lambda x: 1 / (x + 11),
        "ddf": lambda x: -1 / (x + 11) ** 2,
        "interval": (-9.9, 10.0),
        "initial": 0.0,
    },
    "rational": {
        "label": r"$y=1/(x+11)+1$",
        "f": lambda x: 1 / (x + 11) + 1,
        "df": lambda x: -1 / (x + 11) ** 2,
        "ddf": lambda x: 2 / (x + 11) ** 3,
        "interval": (-9.9, 10.0),
        "initial": 0.0,
    },
    "radical": {
        "label": r"$y=\sqrt{x+10}+1$",
        "f": lambda x: np.sqrt(x + 10) + 1,
        "df": lambda x: 1 / (2 * np.sqrt(x + 10)),
        "ddf": lambda x: -1 / (4 * (x + 10) ** 1.5),
        "interval": (-9.9, 10.0),
        "initial": 0.0,
    },
}


def plot_example(name, config):
    """Create separate five-panel plots for Newton and golden-section."""
    f, df, ddf = config["f"], config["df"], config["ddf"]
    a, b = config["interval"]
    newton_figure, newton_axes = plt.subplots(
        2, 3, figsize=(15, 9), constrained_layout=True
    )
    golden_figure, golden_axes = plt.subplots(
        2, 3, figsize=(15, 9), constrained_layout=True
    )
    newton_axes = newton_axes.ravel()
    golden_axes = golden_axes.ravel()
    results = []

    for newton_axis, golden_axis, (x0, y0) in zip(
        newton_axes, golden_axes, POINTS
    ):
        newton = find_distance_newton(
            x0, y0, f, df, ddf, initial_guess=config["initial"]
        )
        golden = golden_section_search(x0, y0, f, a, b)
        nd, nx, ny, nhistory = newton
        gd, gx, gy, ghistory = golden

        curve_x = np.linspace(a, b, 800)
        newton_x = np.array([step["x"] for step in nhistory] + [nx])
        selected = ghistory[::max(1, len(ghistory) // 7)]
        golden_x = np.array([
            step["x1"] if step["D_x1"] < step["D_x2"] else step["x2"]
            for step in selected
        ])

        for axis in (newton_axis, golden_axis):
            axis.plot(curve_x, f(curve_x), color="#17365d", linewidth=2.2,
                      label=config["label"])
            axis.scatter([x0], [y0], marker="*", s=150, color="#d1495b",
                         zorder=6, label=f"Given point ({x0}, {y0})")
            axis.annotate(f"P=({x0}, {y0})", (x0, y0), xytext=(8, 10),
                          textcoords="offset points", fontsize=8, fontweight="bold")
            axis.set_xlim(a, b)
            axis.set_title(f"P = ({x0}, {y0})")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.grid(alpha=0.25)

        newton_axis.plot(
            newton_x, f(newton_x), "o--", color="#f28e2b", markersize=5,
            linewidth=1.4, label="Newton steps"
        )
        newton_axis.plot(
            [x0, nx], [y0, ny], color="#2a9d8f", linewidth=2,
            label=f"Minimum distance = {nd:.4f}"
        )
        newton_axis.scatter([nx], [ny], color="#2a9d8f", s=55, zorder=6)
        for number, x_value in enumerate(newton_x[:-1]):
            newton_axis.annotate(f"k={number}", (x_value, f(x_value)),
                                 xytext=(4, 5), textcoords="offset points", fontsize=7)
        newton_axis.annotate(
            f"Q=({nx:.3f}, {ny:.3f})", (nx, ny), xytext=(8, 10),
            textcoords="offset points", fontsize=8, fontweight="bold",
            color="#166f69", bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8, "ec": "none"}
        )
        newton_axis.legend(fontsize=8, loc="best")

        golden_axis.scatter(
            golden_x, f(golden_x), color="#6f4aa8", s=35, zorder=5,
            label="Golden-section steps"
        )
        golden_axis.plot(
            [x0, gx], [y0, gy], color="#4e79a7", linewidth=2,
            label=f"Minimum distance = {gd:.4f}"
        )
        golden_axis.scatter([gx], [gy], color="#4e79a7", s=55, zorder=6)
        for number, x_value in enumerate(golden_x):
            golden_axis.annotate(f"k={number}", (x_value, f(x_value)),
                                 xytext=(4, 5), textcoords="offset points", fontsize=7)
        golden_axis.annotate(
            f"Q=({gx:.3f}, {gy:.3f})", (gx, gy), xytext=(8, 10),
            textcoords="offset points", fontsize=8, fontweight="bold",
            color="#315f8c", bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8, "ec": "none"}
        )
        golden_axis.legend(fontsize=8, loc="best")

        results.append({
            "function": name,
            "equation": config["label"].replace("$", ""),
            "point": f"({x0}, {y0})",
            "newton_x": nx,
            "newton_y": ny,
            "newton_distance": nd,
            "golden_x": gx,
            "golden_y": gy,
            "golden_distance": gd,
            "difference": abs(nd - gd),
        })

    newton_axes[-1].axis("off")
    golden_axes[-1].axis("off")
    newton_figure.suptitle(
        f"{name.title()} example: Newton-Raphson steps",
        fontsize=18, fontweight="bold"
    )
    golden_figure.suptitle(
        f"{name.title()} example: golden-section steps",
        fontsize=18, fontweight="bold"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    newton_path = OUTPUT_DIR / f"{name}_newton_steps.png"
    golden_path = OUTPUT_DIR / f"{name}_golden_section_steps.png"
    newton_figure.savefig(newton_path, dpi=180, bbox_inches="tight")
    golden_figure.savefig(golden_path, dpi=180, bbox_inches="tight")
    plt.close(newton_figure)
    plt.close(golden_figure)
    print(f"Saved {newton_path}")
    print(f"Saved {golden_path}")
    return results


def main():
    all_results = []
    for name, config in EXAMPLES.items():
        all_results.extend(plot_example(name, config))

    csv_path = OUTPUT_DIR / "non_polynomial_method_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
