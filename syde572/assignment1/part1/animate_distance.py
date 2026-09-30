"""Animate the shortest-distance search from a point to a curve.

Each GIF has three phases:
  1. Slide: a point Q moves along the curve while D(x) is traced below.
  2. Newton-Raphson: the iterates jump toward the minimum of D(x).
  3. Golden-section: the bracket [a, b] shrinks around the minimum.
The final frames show the shortest segment meeting the tangent at a right angle.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from A1 import find_distance_newton
from A2 import golden_section_search


OUTPUT_DIR = Path(__file__).resolve().parent / "media"

CURVE = "#17365d"
POINT = "#d1495b"
NEWTON = "#f28e2b"
GOLDEN = "#7b2cbf"
FINAL = "#2a9d8f"
SLIDE = "#6b7280"


def parabola(a, b, c):
    """Return f, f', f'' for y = a x^2 + b x + c."""
    return (
        lambda x: a * x**2 + b * x + c,
        lambda x: 2 * a * x + b,
        lambda x: 2 * a + 0 * x,
    )


def animate(name, label, f, df, ddf, point, x_range, y_range, d_top,
            newton_start, bracket, mirror_x=None, golden_steps=10, fps=12):
    x0, y0 = point

    def D(x):
        return (x - x0)**2 + (f(x) - y0)**2

    def D_prime(x):
        return 2 * (x - x0) + 2 * (f(x) - y0) * df(x)

    distance, qx, qy, newton_history = find_distance_newton(
        x0, y0, f, df, ddf, initial_guess=newton_start
    )
    _, _, _, golden_history = golden_section_search(x0, y0, f, *bracket)

    # Newton iterates, dropping steps too small to see
    newton_xs = []
    for value in [step["x"] for step in newton_history] + [qx]:
        if not newton_xs or abs(value - newton_xs[-1]) > 1e-4:
            newton_xs.append(value)

    slide_xs = np.linspace(x_range[0] + 0.4, x_range[1] - 0.4, 64)
    frames = (
        [("slide", i) for i in range(len(slide_xs))]
        + [("newton", k) for k in range(len(newton_xs)) for _ in range(9)]
        + [("golden", s) for s in range(min(golden_steps, len(golden_history)))
           for _ in range(5)]
        + [("final", 0)] * 36
    )

    figure = plt.figure(figsize=(7.4, 9.6))
    ax = figure.add_axes([0.11, 0.36, 0.84, 0.58])
    ax_d = figure.add_axes([0.11, 0.06, 0.84, 0.22])

    # Top panel: curve and point
    curve_x = np.linspace(*x_range, 800)
    ax.plot(curve_x, f(curve_x), color=CURVE, linewidth=2.4, label=label)
    ax.scatter([x0], [y0], marker="*", s=240, color=POINT, zorder=6,
               label=f"P = ({x0:g}, {y0:g})")
    ax.annotate("P", (x0, y0), xytext=(9, -4), textcoords="offset points",
                fontsize=12, fontweight="bold", color=POINT)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    best_segment, = ax.plot([], [], color=FINAL, linewidth=1.6, alpha=0.45,
                            linestyle="--", label="_best")
    segment, = ax.plot([], [], color=SLIDE, linewidth=2.2)
    q_dot, = ax.plot([], [], "o", color=SLIDE, markersize=9, zorder=7)
    q_label = ax.annotate("", (0, 0), xytext=(8, 8), textcoords="offset points",
                          fontsize=10, fontweight="bold")
    trail, = ax.plot([], [], "o--", color=NEWTON, markersize=6, linewidth=1.2,
                     zorder=6)
    probes, = ax.plot([], [], "D", color=GOLDEN, markersize=6, zorder=6)
    tangent, = ax.plot([], [], color=FINAL, linewidth=1.2, linestyle=":")
    right_angle, = ax.plot([], [], color=FINAL, linewidth=1.4)
    mirror_segment, = ax.plot([], [], color=FINAL, linewidth=1.8, alpha=0.5,
                              linestyle="--")
    bracket_top = Rectangle((0, 0), 0, 1, transform=ax.get_xaxis_transform(),
                            color=GOLDEN, alpha=0.10, zorder=0)
    ax.add_patch(bracket_top)
    title = ax.set_title("", fontsize=12, fontweight="bold", loc="left")
    info = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                   fontsize=10, family="monospace",
                   bbox={"boxstyle": "round,pad=0.4", "fc": "white",
                         "ec": "#cbd5e1", "alpha": 0.95})

    # Bottom panel: the objective D(x)
    ax_d.plot(curve_x, D(curve_x), color=CURVE, linewidth=2)
    ax_d.set_xlim(*x_range)
    ax_d.set_ylim(0, d_top)
    ax_d.grid(alpha=0.25)
    ax_d.set_xlabel("x")
    ax_d.set_ylabel(r"$D(x)=(x-x_0)^2+(f(x)-y_0)^2$", fontsize=9)
    cursor = ax_d.axvline(slide_xs[0], color=SLIDE, linewidth=1, alpha=0.6)
    d_dot, = ax_d.plot([], [], "o", color=SLIDE, markersize=8, zorder=5)
    d_trail, = ax_d.plot([], [], "o--", color=NEWTON, markersize=5,
                         linewidth=1.1, zorder=5)
    d_probes, = ax_d.plot([], [], "D", color=GOLDEN, markersize=5, zorder=5)
    bracket_bottom = Rectangle((0, 0), 0, 1,
                               transform=ax_d.get_xaxis_transform(),
                               color=GOLDEN, alpha=0.12, zorder=0)
    ax_d.add_patch(bracket_bottom)

    # Line the bottom panel up with the equal-aspect top panel
    figure.canvas.draw()
    top = ax.get_position()
    bottom = ax_d.get_position()
    ax_d.set_position([top.x0, bottom.y0, top.width, bottom.height])

    def place_q(x, color):
        y = f(x)
        segment.set_data([x0, x], [y0, y])
        segment.set_color(color)
        q_dot.set_data([x], [y])
        q_dot.set_color(color)
        q_label.xy = (x, y)
        q_label.set_text("Q")
        q_label.set_color(color)
        cursor.set_xdata([x, x])
        cursor.set_color(color)
        d_dot.set_data([x], [D(x)])
        d_dot.set_color(color)

    def reset():
        for artist in (best_segment, trail, probes, tangent, right_angle,
                       mirror_segment, d_trail, d_probes):
            artist.set_data([], [])
        bracket_top.set_width(0)
        bracket_bottom.set_width(0)

    def update(frame):
        phase, index = frame
        reset()

        if phase == "slide":
            x = slide_xs[index]
            place_q(x, SLIDE)
            seen = slide_xs[:index + 1]
            best = seen[np.argmin(D(seen))]
            best_segment.set_data([x0, best], [y0, f(best)])
            title.set_text("1 · Slide Q along the curve")
            info.set_text(
                f"Q = ({x:6.3f}, {f(x):6.3f})\n"
                f"|PQ|      = {math.sqrt(D(x)):7.4f}\n"
                f"best seen = {math.sqrt(D(best)):7.4f}"
            )

        elif phase == "newton":
            x = newton_xs[index]
            shown = np.array(newton_xs[:index + 1])
            trail.set_data(shown, f(shown))
            d_trail.set_data(shown, D(shown))
            place_q(x, NEWTON)
            title.set_text("2 · Newton–Raphson on D'(x) = 0")
            info.set_text(
                f"k = {index}\n"
                f"x_k   = {x:9.6f}\n"
                f"D'(x) = {D_prime(x):9.6f}\n"
                f"|PQ|  = {math.sqrt(D(x)):9.6f}"
            )

        elif phase == "golden":
            step = golden_history[index]
            a, b, x1, x2 = step["a"], step["b"], step["x1"], step["x2"]
            for rect in (bracket_top, bracket_bottom):
                rect.set_x(a)
                rect.set_width(b - a)
            probes.set_data([x1, x2], [f(x1), f(x2)])
            d_probes.set_data([x1, x2], [step["D_x1"], step["D_x2"]])
            place_q((a + b) / 2, GOLDEN)
            title.set_text("3 · Golden-section search shrinks [a, b]")
            info.set_text(
                f"k = {index}\n"
                f"[a, b] = [{a:7.4f}, {b:7.4f}]\n"
                f"width  = {b - a:.4f}"
            )

        else:
            place_q(qx, FINAL)
            slope = df(qx)
            t = np.array([1.0, slope]) / math.hypot(1.0, slope)
            n = np.array([x0 - qx, y0 - qy]) / distance
            tangent.set_data([qx - 1.6 * t[0], qx + 1.6 * t[0]],
                             [qy - 1.6 * t[1], qy + 1.6 * t[1]])
            s = 0.32
            corner = [np.array([qx, qy]) + s * t,
                      np.array([qx, qy]) + s * (t + n),
                      np.array([qx, qy]) + s * n]
            right_angle.set_data([p[0] for p in corner],
                                 [p[1] for p in corner])
            lines = [
                f"Q* = ({qx:.5f}, {qy:.5f})",
                f"d  = {distance:.5f}",
                "PQ* ⟂ tangent at Q*",
            ]
            if mirror_x is not None:
                mirror_segment.set_data([x0, mirror_x], [y0, f(mirror_x)])
                lines.append("dashed: equal minimum")
            title.set_text("Shortest distance found")
            info.set_text("\n".join(lines))

        return ()

    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    output_path = OUTPUT_DIR / f"{name}.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(figure)
    print(f"Saved {output_path} (d = {distance:.5f} at x = {qx:.5f})")


if __name__ == "__main__":
    # Supplied parabola and point
    f, df, ddf = parabola(1, 0, 5)
    animate(
        "distance_search_parabola",
        r"$y=x^2+5$",
        f, df, ddf,
        point=(-4, 0),
        x_range=(-6.5, 3.5),
        y_range=(-1.5, 8.5),
        d_top=160,
        newton_start=2.5,
        bracket=(-6, 3),
    )

    # Another parabola with the point above the vertex: two equal minima
    f, df, ddf = parabola(0.5, 0, 1)
    animate(
        "distance_search_two_minima",
        r"$y=0.5x^2+1$",
        f, df, ddf,
        point=(0, 5),
        x_range=(-4.5, 4.5),
        y_range=(-0.5, 8.5),
        d_top=30,
        newton_start=3.5,
        bracket=(0, 4.5),
        mirror_x=-math.sqrt(6),
    )
