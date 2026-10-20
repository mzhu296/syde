"""Part 2.1: image sharpening with the unsharp mask filter."""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from part1_2 import HERE, OUT, load_gray
from part1_3 import gaussian2d

TAJ = HERE / "syde671 assn2 part 2 images" / "taj.jpg"
SELFIE = HERE / "data" / "IMG_7415.jpg"


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=float) / 255


def blur(img, G):
    """Gaussian blur, per channel for colour images (symmetric boundary: no dark border)."""
    if img.ndim == 2:
        return convolve2d(img, G, mode="same", boundary="symm")
    return np.dstack([blur(img[..., c], G) for c in range(img.shape[2])])


def unsharp_kernel(G, alpha):
    """Single-convolution unsharp mask: (1 + alpha) * impulse - alpha * G."""
    impulse = np.zeros_like(G)
    impulse[G.shape[0] // 2, G.shape[1] // 2] = 1.0
    return (1 + alpha) * impulse - alpha * G


def sharpen(img, G, alpha):
    """Apply the single unsharp kernel (clipped to the valid display range)."""
    K = unsharp_kernel(G, alpha)
    if img.ndim == 2:
        out = convolve2d(img, K, mode="same", boundary="symm")
    else:
        out = np.dstack([convolve2d(img[..., c], K, mode="same", boundary="symm") for c in range(img.shape[2])])
    return np.clip(out, 0, 1)


def show(ax, im, title):
    ax.imshow(np.clip(im, 0, 1), cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def mse(a, b):
    return float(np.mean((a - b) ** 2))


if __name__ == "__main__":
    G = gaussian2d(13, 2.0)
    ALPHA = 1.5

    # ---- 1. blurred / high-frequency / sharpened for two images ----
    images = {"Taj Mahal": load_rgb(TAJ), "selfie": load_rgb(SELFIE, 512)}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    for row, (name, img) in zip(axes, images.items()):
        bl = blur(img, G)
        high = img - bl
        sh = sharpen(img, G, ALPHA)
        # verify the single kernel equals the three-step recipe img + alpha * (img - blur)
        three = np.clip(img + ALPHA * high, 0, 1)
        m = 13
        print(f"{name}: single-kernel vs three-step max diff  interior {np.abs(sh - three)[m:-m, m:-m].max():.1e}"
              f"   whole image {np.abs(sh - three).max():.1e}")
        show(row[0], img, f"{name}: original")
        show(row[1], bl, "blurred (low frequencies)")
        show(row[2], 0.5 + high, "high frequencies (img - blur, +0.5)")
        show(row[3], sh, f"sharpened, alpha = {ALPHA}")
    fig.tight_layout(); fig.savefig(OUT / "part2_1_steps.png", dpi=110); plt.close(fig)

    # ---- 2. effect of the sharpening amount alpha (Taj) ----
    taj = images["Taj Mahal"]
    alphas = [0, 0.5, 1, 2, 4, 8]
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.6 * len(alphas), 4))
    for ax, a in zip(axes, alphas):
        show(ax, sharpen(taj, G, a), f"alpha = {a}")
    fig.tight_layout(); fig.savefig(OUT / "part2_1_alpha.png", dpi=110); plt.close(fig)

    # ---- 3. effect of the Gaussian width sigma at fixed alpha (Taj) ----
    sigmas = [0.5, 1, 2, 4, 8]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(3.6 * len(sigmas), 4))
    for ax, s in zip(axes, sigmas):
        k = int(6 * s) | 1  # odd kernel size ~ 6 sigma
        show(ax, sharpen(taj, gaussian2d(max(k, 3), s), ALPHA), f"sigma = {s}")
    fig.tight_layout(); fig.savefig(OUT / "part2_1_sigma.png", dpi=110); plt.close(fig)

    # ---- 4. evaluation: sharp image -> blur -> sharpen again ----
    sharp = load_gray("cameraman.png")
    blurred = blur(sharp, gaussian2d(13, 2.0))
    print(f"\nevaluation (cameraman): MSE(original, blurred) = {mse(sharp, blurred):.5f}")
    sweep = np.linspace(0, 6, 61)
    errs = [mse(sharp, sharpen(blurred, G, a)) for a in sweep]
    best = sweep[int(np.argmin(errs))]
    print(f"  best alpha by MSE = {best:.1f}: MSE = {min(errs):.5f}   (alpha = 0 gives {errs[0]:.5f})")
    resharp = sharpen(blurred, G, best)

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    show(axes[0], sharp, "original (sharp)")
    show(axes[1], blurred, f"blurred, sigma = 2 (MSE {mse(sharp, blurred):.4f})")
    show(axes[2], resharp, f"re-sharpened, alpha = {best:.1f} (MSE {min(errs):.4f})")
    show(axes[3], 0.5 + (sharp - resharp), "original - re-sharpened (+0.5)")
    axes[4].plot(sweep, errs); axes[4].axvline(best, ls="--", c="gray")
    axes[4].set_xlabel("alpha"); axes[4].set_ylabel("MSE vs original"); axes[4].set_title("error vs sharpening amount")
    fig.tight_layout(); fig.savefig(OUT / "part2_1_eval.png", dpi=110); plt.close(fig)
    print("saved figures to", OUT)
