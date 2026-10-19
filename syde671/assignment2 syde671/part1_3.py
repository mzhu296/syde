"""Part 1.3: derivative of Gaussian (DoG) filters, plus gradient-orientation bell & whistle."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d

from part1_2 import Dx, Dy, OUT, gradients, load_gray

KSIZE, SIGMA = 13, 2.0  # ksize ~ 6*sigma+1 so the kernel is not truncated


def gaussian2d(ksize=KSIZE, sigma=SIGMA):
    g = cv2.getGaussianKernel(ksize, sigma)  # (ksize, 1) column vector
    return g @ g.T  # outer product -> 2D kernel, sums to 1


def my_atan2(y, x):
    """Angle of (x, y) in (-pi, pi], without built-in angle functions.

    arctan(z) on [0, 1] uses the polynomial of Abramowitz & Stegun 4.4.49 (max error about 1e-5 rad);
    the octant / quadrant is then recovered with abs, comparisons and reflections.
    """
    ax, ay = np.abs(x), np.abs(y)
    swap = ay > ax                                   # keep the ratio z = small / large <= 1
    num, den = np.where(swap, ax, ay), np.where(swap, ay, ax)
    z = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
    z2 = z * z
    a = z * (0.9998660 + z2 * (-0.3302995 + z2 * (0.1801410 + z2 * (-0.0851330 + z2 * 0.0208351))))
    a = np.where(swap, np.pi / 2 - a, a)             # angle in the first quadrant, [0, pi/2]
    a = np.where(x < 0, np.pi - a, a)                # reflect across the y axis
    return np.where(y < 0, -a, a)                    # reflect across the x axis


def orientation_hsv(gx, gy):
    """Hue = gradient direction, saturation = 1, value = normalized magnitude."""
    ang = (my_atan2(gy, gx) % (2 * np.pi)) / (2 * np.pi)
    mag = np.hypot(gx, gy)
    hsv = np.dstack([ang, np.ones_like(ang), mag / mag.max()])
    return hsv_to_rgb(hsv)


def colorwheel(n=200):
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]  # y grows downward on screen, like image rows
    r = np.hypot(x, y)
    hsv = np.dstack([(my_atan2(y, x) % (2 * np.pi)) / (2 * np.pi), np.ones_like(r), (r <= 1).astype(float)])
    return hsv_to_rgb(hsv)


if __name__ == "__main__":
    img = load_gray("cameraman.png")
    G = gaussian2d()

    # --- two-step: blur, then finite differences (same procedure as 1.2) ---
    blurred = convolve2d(img, G, mode="same", boundary="symm")
    gx, gy, mag = gradients(blurred)

    # --- one-step: DoG filters = Gaussian convolved with Dx / Dy ---
    DoGx = convolve2d(G, Dx, mode="full")
    DoGy = convolve2d(G, Dy, mode="full")
    gx2 = convolve2d(img, DoGx, mode="same", boundary="symm")
    gy2 = convolve2d(img, DoGy, mode="same", boundary="symm")
    mag2 = np.hypot(gx2, gy2)

    # verification: exact with full convolution (associativity); near-exact away from borders for 'same'
    full_two = convolve2d(convolve2d(img, G, "full"), Dx, "full")
    full_one = convolve2d(img, DoGx, "full")
    m = KSIZE  # ignore a border of about one kernel width where boundary handling differs
    print("full-mode max |two-step - one-step|     : %.2e" % np.abs(full_two - full_one).max())
    print("same-mode interior max diff (gx, gy)    : %.2e, %.2e"
          % (np.abs(gx - gx2)[m:-m, m:-m].max(), np.abs(gy - gy2)[m:-m, m:-m].max()))
    print("same-mode whole-image max diff (gx, gy) : %.2e, %.2e"
          % (np.abs(gx - gx2).max(), np.abs(gy - gy2).max()))
    print("blurred gradient magnitude: max %.3f  99th pct %.3f" % (mag.max(), np.percentile(mag, 99)))

    # threshold sweep on the blurred gradient magnitude
    sweep = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    fig, axes = plt.subplots(1, len(sweep), figsize=(3 * len(sweep), 3.3))
    for ax, t in zip(axes, sweep):
        ax.imshow(mag > t, cmap="gray"); ax.set_title(f"threshold {t:.2f}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "part1_3_threshold_sweep.png", dpi=120); plt.close(fig)

    THRESH = 0.05  # chosen from the sweep: all real edges kept, no grass noise
    edges = mag > THRESH

    # Figure 1: blurred pipeline, compared with the un-blurred result from 1.2
    _, _, mag_raw = gradients(img)
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.4))
    panels = [(blurred, "Gaussian-blurred"), (gx, "Dx of blurred"), (gy, "Dy of blurred"),
              (mag, "gradient magnitude (blurred)"), (edges, f"edges, threshold {THRESH}"),
              (mag_raw > 0.20, "1.2: no blur, threshold 0.20")]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray"); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "part1_3_blur_pipeline.png", dpi=120); plt.close(fig)

    # Figure 2: DoG filters, single-convolution result, and difference from the two-step result
    edges2 = mag2 > THRESH
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.4))
    panels = [(DoGx, "DoG_x filter"), (DoGy, "DoG_y filter"), (gx2, "DoG_x * image"),
              (gy2, "DoG_y * image"), (edges2, f"edges (one step), threshold {THRESH}"),
              (np.abs(gx - gx2) + np.abs(gy - gy2), "")]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray", **(dict(vmin=0, vmax=1e-2) if title == "" else {}))
        ax.set_title(title); ax.axis("off")
    axes[5].set_title("|two-step - one-step|, gx+gy\nmax %.1e (scale 0 to 1e-2)" % (np.abs(gx - gx2) + np.abs(gy - gy2)).max())
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(OUT / "part1_3_dog.png", dpi=120); plt.close(fig)
    print("edge maps identical (one-step vs two-step):", np.array_equal(edges, edges2))

    # Bell & whistle: gradient orientation in HSV (hue = direction, brightness = magnitude)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("original")
    axes[1].imshow(orientation_hsv(gx2, gy2)); axes[1].set_title("gradient orientation (hue) x magnitude (value)")
    axes[2].imshow(colorwheel()); axes[2].set_title("hue = direction of increasing intensity\n(right = red, down = yellow-green)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "part1_3_orientation.png", dpi=120); plt.close(fig)
    print("saved figures to", OUT)
