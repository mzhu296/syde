"""Part 1.2: finite difference operator on the cameraman image."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

Dx = np.array([[1, -1]], dtype=float)
Dy = np.array([[1], [-1]], dtype=float)


def load_gray(name):
    return np.asarray(Image.open(HERE / "data" / name).convert("L"), dtype=float) / 255


def gradients(img):
    # symmetric boundary: zero fill would create a fake bright edge along the image border
    gx = convolve2d(img, Dx, mode="same", boundary="symm")
    gy = convolve2d(img, Dy, mode="same", boundary="symm")
    return gx, gy, np.hypot(gx, gy)


if __name__ == "__main__":
    img = load_gray("cameraman.png")
    gx, gy, mag = gradients(img)
    print("gradient magnitude: min %.3f  max %.3f  mean %.3f  99th pct %.3f"
          % (mag.min(), mag.max(), mag.mean(), np.percentile(mag, 99)))

    # threshold sweep, used to choose the final value by eye
    sweep = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    fig, axes = plt.subplots(1, len(sweep), figsize=(3 * len(sweep), 3.3))
    for ax, t in zip(axes, sweep):
        ax.imshow(mag > t, cmap="gray")
        ax.set_title(f"threshold {t:.2f}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "part1_2_threshold_sweep.png", dpi=120)

    THRESH = 0.20  # chosen qualitatively from the sweep: keeps tripod/camera/coat edges, little grass noise
    edges = mag > THRESH

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
    panels = [
        (img, "original", dict(cmap="gray")),
        (gx, "partial derivative in x (Dx)", dict(cmap="gray")),
        (gy, "partial derivative in y (Dy)", dict(cmap="gray")),
        (mag, "gradient magnitude", dict(cmap="gray")),
        (edges, f"binarized edges (threshold {THRESH})", dict(cmap="gray")),
    ]
    for ax, (im, title, kw) in zip(axes, panels):
        ax.imshow(im, **kw)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "part1_2.png", dpi=120)
    print("saved", OUT / "part1_2.png")
