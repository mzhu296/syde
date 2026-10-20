"""Part 2.3: Gaussian and Laplacian stacks (no downsampling, no built-in pyramid functions)."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d

from part1_2 import HERE, OUT
from part2_1 import load_rgb

APPLE = HERE / "syde671 assn2 part 2 images" / "burt_apple.png"
ORANGE = HERE / "syde671 assn2 part 2 images" / "burt_orange.png"

# Gaussian widths of the stack levels: level 0 is the image itself, then sigma doubles per level,
# which mimics a pyramid whose resolution halves at every level.
SIGMAS = [0, 2, 4, 8, 16, 32]


def blur(img, sigma):
    """Separable Gaussian blur on a 2D or HxWxC array, reflecting at the border."""
    if sigma == 0:
        return img.copy()
    k = cv2.getGaussianKernel(int(6 * sigma) | 1, sigma).ravel()
    return convolve1d(convolve1d(img, k, axis=0, mode="reflect"), k, axis=1, mode="reflect")


def gaussian_stack(img, sigmas=SIGMAS):
    return [blur(img, s) for s in sigmas]


def laplacian_stack(img, sigmas=SIGMAS):
    """Band-pass levels G_i - G_{i+1}; the last level is the remaining low-pass image, so the levels sum to img."""
    g = gaussian_stack(img, sigmas)
    return [g[i] - g[i + 1] for i in range(len(g) - 1)] + [g[-1]]


def blend(a, b, mask, sigmas=SIGMAS):
    """Burt & Adelson multiresolution blend using stacks: sum_i GM_i * LA_i + (1 - GM_i) * LB_i."""
    return sum(gm * la + (1 - gm) * lb
               for gm, la, lb in zip(gaussian_stack(mask, sigmas), laplacian_stack(a, sigmas), laplacian_stack(b, sigmas)))


def masked_levels(a, b, mask, sigmas=SIGMAS):
    """Per-level contributions GM_i * LA_i and (1 - GM_i) * LB_i (the columns of Szeliski Fig. 3.42)."""
    gms = gaussian_stack(mask, sigmas)
    la, lb = laplacian_stack(a, sigmas), laplacian_stack(b, sigmas)
    return [gm * x for gm, x in zip(gms, la)], [(1 - gm) * y for gm, y in zip(gms, lb)]


def band(x):
    """Show a signed band-pass level around mid-gray."""
    return np.clip(0.5 + x, 0, 1)


def step_mask(shape, axis=1, frac=0.5):
    """Left (axis=1) or top (axis=0) fraction of the image is 1: a vertical or horizontal seam."""
    m = np.zeros(shape[:2])
    n = int(round(frac * shape[axis]))
    if axis == 1:
        m[:, :n] = 1
    else:
        m[:n, :] = 1
    return m[..., None] if len(shape) == 3 else m


def process_figure(a, b, mask, path, rows=(0, 2, 4), sigmas=SIGMAS):
    """Szeliski Fig. 3.42 / Burt & Adelson Fig. 10: weighted bands of each input, their sum, then collapsed images."""
    ma, mb = masked_levels(a, b, mask, sigmas)
    fig, axes = plt.subplots(len(rows) + 1, 3, figsize=(10.5, 3.3 * (len(rows) + 1) * a.shape[0] / a.shape[1] * 1.3))
    for r, lvl in enumerate(rows):
        for c, im in enumerate([ma[lvl], mb[lvl], ma[lvl] + mb[lvl]]):
            axes[r, c].imshow(band(im)); axes[r, c].axis("off")
            axes[r, c].text(0.5, -0.03, f"({'abcdefghijkl'[3 * r + c]})", transform=axes[r, c].transAxes, ha="center", va="top")
        axes[r, 0].set_title(f"level {lvl} (sigma {sigmas[lvl]} to {sigmas[lvl + 1] if lvl + 1 < len(sigmas) else 'residual'})",
                             fontsize=9, loc="left")
    total_a, total_b = sum(ma), sum(mb)
    for c, im in enumerate([total_a, total_b, total_a + total_b]):
        axes[-1, c].imshow(np.clip(im, 0, 1)); axes[-1, c].axis("off")
        axes[-1, c].text(0.5, -0.03, f"({'abcdefghijkl'[3 * len(rows) + c]})", transform=axes[-1, c].transAxes, ha="center", va="top")
    axes[0, 0].text(0.5, 1.28, "A x mask", transform=axes[0, 0].transAxes, ha="center")
    axes[0, 1].text(0.5, 1.28, "B x (1 - mask)", transform=axes[0, 1].transAxes, ha="center")
    axes[0, 2].text(0.5, 1.28, "sum", transform=axes[0, 2].transAxes, ha="center")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def stack_figure(img, mask, path, sigmas=SIGMAS, title=""):
    """Gaussian, Laplacian and mask stacks for one image, one column per level."""
    g, l, gm = gaussian_stack(img, sigmas), laplacian_stack(img, sigmas), gaussian_stack(mask, sigmas)
    n = len(sigmas)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 6.4))
    for i in range(n):
        axes[0, i].imshow(np.clip(g[i], 0, 1)); axes[0, i].set_title(f"G{i}: sigma {sigmas[i]}", fontsize=9)
        axes[1, i].imshow(np.clip(g[i], 0, 1) if i == n - 1 else band(l[i]))
        axes[1, i].set_title(f"L{i}" + (" (residual = G%d)" % i if i == n - 1 else f" = G{i} - G{i + 1}"), fontsize=9)
        axes[2, i].imshow(gm[i].squeeze(), cmap="gray", vmin=0, vmax=1); axes[2, i].set_title(f"mask G{i}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


if __name__ == "__main__":
    apple, orange = load_rgb(APPLE), load_rgb(ORANGE)
    mask = step_mask(apple.shape)

    # sanity checks
    rec = sum(laplacian_stack(apple))
    print("Laplacian stack sums back to the image, max error: %.1e" % np.abs(rec - apple).max())
    print("blend(A, A, mask) == A, max error:              %.1e" % np.abs(blend(apple, apple, mask) - apple).max())
    print("blend with all-ones mask == A, max error:       %.1e" % np.abs(blend(apple, orange, np.ones_like(mask)) - apple).max())

    stack_figure(apple, mask, OUT / "part2_3_stack_apple.png", title="Apple: Gaussian stack, Laplacian stack, mask stack")
    stack_figure(orange, 1 - mask, OUT / "part2_3_stack_orange.png", title="Orange: Gaussian stack, Laplacian stack, (1 - mask) stack")
    process_figure(apple, orange, mask, OUT / "part2_3_fig342.png")
    print("saved part 2.3 figures to", OUT)
