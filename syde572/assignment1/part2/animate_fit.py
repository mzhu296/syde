"""Animate coordinate-wise Newton-Raphson fitting for Part Two.

Every frame is one single-coefficient update, so the path on the MSE
contours moves along one axis at a time. Early sweeps are shown step by
step; later sweeps are sampled so the whole run fits in a short loop.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from part2_newton import X_DATA, Y_DATA, coordinate_newton


OUTPUT_DIR = Path(__file__).resolve().parent / "media"

DATA = "#d1495b"
FIT = "#f28e2b"
PATH = "#17365d"
FINAL = "#2a9d8f"
NAMES = ["c₀", "c₁", "c₂"]


def coordinate_path(design_matrix, targets, sweeps):
    """Coefficients after every single-parameter Newton update."""
    coefficients = np.zeros(design_matrix.shape[1])
    states = [(0, None, coefficients.copy())]
    for sweep in range(1, sweeps + 1):
        for parameter in range(design_matrix.shape[1]):
            residual = design_matrix @ coefficients - targets
            column = design_matrix[:, parameter]
            coefficients[parameter] -= (column @ residual) / (column @ column)
            states.append((sweep, parameter, coefficients.copy()))
    return states


def mse(design_matrix, coefficients):
    residual = design_matrix @ coefficients - Y_DATA
    return float(residual @ residual) / len(Y_DATA)


def build_frames(states, n_params, detailed_sweeps, sampled_sweeps):
    """Frame indices into states: every update early, sampled sweeps later."""
    frames = [0] * 8
    for index, (sweep, _, _) in enumerate(states):
        if 1 <= sweep <= detailed_sweeps:
            frames += [index] * 6
    for sweep in sampled_sweeps:
        frames += [sweep * n_params] * 7
    frames += [len(states) - 1] * 30
    return frames


def animate(name, powers, contour_axes, contour_box, sampled_sweeps,
            detailed_sweeps=8):
    design_matrix = np.column_stack([X_DATA**p for p in powers])
    n_params = len(powers)
    solution, history = coordinate_newton(design_matrix, Y_DATA)
    total_sweeps = len(history) - 1
    states = coordinate_path(design_matrix, Y_DATA, total_sweeps)
    assert np.allclose(states[-1][2], solution)
    frames = build_frames(states, n_params, detailed_sweeps,
                          [s for s in sampled_sweeps if s <= total_sweeps])

    show_mse_panel = n_params > 2
    figure, axes = plt.subplots(
        1, 3 if show_mse_panel else 2,
        figsize=(14.5, 4.9) if show_mse_panel else (11, 4.9),
    )
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.86,
                           wspace=0.28)
    fit_axis, contour_axis = axes[0], axes[1]

    # Left: data and current fit
    plot_x = np.linspace(-0.25, 3.25, 300)
    plot_design = np.column_stack([plot_x**p for p in powers])
    fit_axis.plot(plot_x, plot_design @ solution, color=FINAL, linewidth=1.4,
                  linestyle="--", alpha=0.7, label="Analytical fit")
    fit_axis.scatter(X_DATA, Y_DATA, marker="*", s=200, color=DATA, zorder=6,
                     label="Data")
    fit_line, = fit_axis.plot([], [], color=FIT, linewidth=2.6,
                              label="Current fit")
    residual_lines = [fit_axis.plot([], [], color=DATA, linewidth=1,
                                    alpha=0.6)[0] for _ in X_DATA]
    fit_axis.set_xlim(-0.25, 3.25)
    fit_axis.set_ylim(-1, 9)
    fit_axis.set_xlabel("x")
    fit_axis.set_ylabel("y")
    fit_axis.grid(alpha=0.25)
    fit_axis.legend(loc="upper left", fontsize=9)
    info = fit_axis.text(0.97, 0.04, "", transform=fit_axis.transAxes,
                         ha="right", va="bottom", fontsize=9.5,
                         family="monospace",
                         bbox={"boxstyle": "round,pad=0.4", "fc": "white",
                               "ec": "#cbd5e1", "alpha": 0.95})

    # Middle: MSE contours over two coefficients
    i, j = contour_axes
    (lo_i, hi_i), (lo_j, hi_j) = contour_box
    grid_i, grid_j = np.meshgrid(np.linspace(lo_i, hi_i, 220),
                                 np.linspace(lo_j, hi_j, 220))
    surface = np.zeros_like(grid_i)
    others = [k for k in range(n_params) if k not in (i, j)]
    for row in range(grid_i.shape[0]):
        for col in range(grid_i.shape[1]):
            c = np.zeros(n_params)
            c[i], c[j] = grid_i[row, col], grid_j[row, col]
            if others:
                # Profile out the hidden coefficient at its best value
                k = others[0]
                rest = Y_DATA - design_matrix @ c
                c[k] = design_matrix[:, k] @ rest / (
                    design_matrix[:, k] @ design_matrix[:, k])
            surface[row, col] = mse(design_matrix, c)
    levels = np.geomspace(surface.min() * 1.05, surface.max(), 16)
    contour_axis.contourf(grid_i, grid_j, surface, levels=levels,
                          cmap="Blues_r", alpha=0.55)
    contour_axis.contour(grid_i, grid_j, surface, levels=levels,
                         colors="#475569", linewidths=0.5, alpha=0.6)
    contour_axis.scatter([solution[i]], [solution[j]], marker="*", s=220,
                         color=FINAL, zorder=6, label="Analytical minimum")
    path_line, = contour_axis.plot([], [], "-", color=PATH, linewidth=1.4)
    path_head, = contour_axis.plot([], [], "o", color=FIT, markersize=8,
                                   zorder=7, label="Current coefficients")
    contour_axis.set_xlim(lo_i, hi_i)
    contour_axis.set_ylim(lo_j, hi_j)
    contour_axis.set_xlabel(NAMES[i])
    contour_axis.set_ylabel(NAMES[j])
    contour_title = "MSE contours"
    if others:
        contour_title += f" ({NAMES[others[0]]} at its best value)"
    contour_axis.set_title(contour_title, fontsize=10)
    contour_axis.legend(loc="upper right", fontsize=8.5)

    # Right (parabola only): MSE against sweep
    if show_mse_panel:
        mse_axis = axes[2]
        sweeps = np.array([h["iteration"] for h in history[1:]])
        values = np.array([h["mse"] for h in history[1:]])
        mse_axis.loglog(sweeps, values, color="#cbd5e1", linewidth=2)
        mse_axis.axhline(history[-1]["mse"], color=FINAL, linestyle="--",
                         linewidth=1.2, label=f"Final MSE = {history[-1]['mse']:.4f}")
        mse_trace, = mse_axis.loglog([], [], color=FIT, linewidth=2.4)
        mse_dot, = mse_axis.loglog([], [], "o", color=FIT, markersize=7)
        mse_axis.set_xlabel("Sweep")
        mse_axis.set_ylabel("MSE (log scale)")
        mse_axis.set_title("MSE convergence", fontsize=10)
        mse_axis.grid(alpha=0.25, which="both")
        mse_axis.legend(loc="upper right", fontsize=8.5)

    title = figure.suptitle("", fontsize=13, fontweight="bold", x=0.06,
                            ha="left")

    def update(index):
        sweep, parameter, c = states[index]
        fit_line.set_data(plot_x, plot_design @ c)
        predictions = design_matrix @ c
        for line, x, y, p in zip(residual_lines, X_DATA, Y_DATA, predictions):
            line.set_data([x, x], [y, p])

        # Staircase path up to this update
        shown = np.array([s[2] for s in states[:index + 1]])
        path_line.set_data(shown[:, i], shown[:, j])
        path_head.set_data([c[i]], [c[j]])

        error = mse(design_matrix, c)
        if parameter is None:
            action = "start at zero"
        elif index == len(states) - 1:
            action = "converged"
        else:
            held = ", ".join(NAMES[k] for k in range(n_params) if k != parameter)
            action = f"update {NAMES[parameter]} ({held} fixed)"
        coefficient_text = "\n".join(
            f"{NAMES[k]} = {c[k]:8.5f}" for k in range(n_params))
        info.set_text(f"sweep {sweep}\n{coefficient_text}\nMSE = {error:.5f}")
        title.set_text(f"{name.title()} · coordinate-wise Newton–Raphson · "
                       f"{action}")

        if show_mse_panel:
            done = sweeps[sweeps <= max(sweep, 1)]
            mse_trace.set_data(done, values[:len(done)])
            mse_dot.set_data([max(sweep, 1)], [values[max(sweep, 1) - 1]])
        return ()

    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    output_path = OUTPUT_DIR / f"part2_{name}_newton.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=72)
    plt.close(figure)
    print(f"Saved {output_path} ({total_sweeps} sweeps, "
          f"coefficients {np.round(solution, 5)}, MSE {history[-1]['mse']:.5f})")


if __name__ == "__main__":
    animate(
        "line",
        powers=[0, 1],
        contour_axes=(0, 1),
        contour_box=((-1.2, 3.8), (-0.3, 2.9)),
        sampled_sweeps=[12, 18, 26, 36, 55],
    )
    animate(
        "parabola",
        powers=[0, 1, 2],
        contour_axes=(1, 2),
        contour_box=((-0.4, 1.8), (-0.1, 0.95)),
        sampled_sweeps=[12, 20, 35, 60, 100, 170, 300, 591],
    )
