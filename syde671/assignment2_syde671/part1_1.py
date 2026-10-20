"""Part 1.1: convolution from scratch (numpy only) vs scipy.signal.convolve2d."""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def pad_zero(img, kh, kw):
    """Zero-pad so a (kh, kw) kernel yields a same-size output (works for even sizes)."""
    H, W = img.shape
    out = np.zeros((H + kh - 1, W + kw - 1))
    out[kh // 2 : kh // 2 + H, kw // 2 : kw // 2 + W] = img
    return out


def conv_4loops(img, k):
    kh, kw = k.shape
    kf = k[::-1, ::-1]  # convolution flips the kernel
    p = pad_zero(img, kh, kw)
    out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            s = 0.0
            for u in range(kh):
                for v in range(kw):
                    s += p[i + u, j + v] * kf[u, v]
            out[i, j] = s
    return out


def conv_2loops(img, k):
    kh, kw = k.shape
    kf = k[::-1, ::-1]
    p = pad_zero(img, kh, kw)
    out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(p[i : i + kh, j : j + kw] * kf)
    return out


def timed(fn, *args):
    t = time.perf_counter()
    r = fn(*args)
    return r, time.perf_counter() - t


if __name__ == "__main__":
    pil = Image.open(HERE / "data" / "IMG_7415.jpg").convert("L").resize((256, 256), Image.LANCZOS)
    img = np.asarray(pil, dtype=float) / 255

    kernels = {
        "9x9 box": np.ones((9, 9)) / 81,
        "Dx": np.array([[1, -1]], dtype=float),
        "Dy": np.array([[1], [-1]], dtype=float),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("grayscale")

    print(f"{'kernel':8s} {'4 loops':>9s} {'2 loops':>9s} {'scipy':>9s}  match(4)  match(2)  max|err|")
    for ax, (name, k) in zip(axes[1:], kernels.items()):
        r4, t4 = timed(conv_4loops, img, k)
        r2, t2 = timed(conv_2loops, img, k)
        ref, ts = timed(convolve2d, img, k, "same", "fill", 0)
        err = max(np.abs(r4 - ref).max(), np.abs(r2 - ref).max())
        print(f"{name:8s} {t4:8.2f}s {t2:8.2f}s {ts:8.4f}s  {np.allclose(r4, ref)!s:8s}  {np.allclose(r2, ref)!s:8s}  {err:.1e}")

        if name == "9x9 box":
            ax.imshow(r2, cmap="gray")
        else:  # signed derivative: show magnitude of response
            ax.imshow(np.abs(r2), cmap="gray")
        ax.set_title(name if name == "9x9 box" else f"|{name}|")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "part1_1.png", dpi=150)
    print("saved", OUT / "part1_1.png")
