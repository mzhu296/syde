"""Part 2.4: multiresolution blending (the Oraple and custom blends)."""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from part1_2 import HERE, OUT
from part2_1 import load_rgb
from part2_2 import PAIRS, align, gray
from part2_3 import APPLE, ORANGE, SIGMAS, blend, blur, process_figure, step_mask

DATA = HERE / "data"
CARDS = HERE / "syde671 assn2 part 2 images"


def ellipse_mask(shape, center, axes, angle=0.0, value=1.0):
    m = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(m, tuple(int(v) for v in center), tuple(int(v) for v in axes), angle, 0, 360, 255, -1)
    return (m / 255.0 * value)[..., None] if len(shape) == 3 else m / 255.0 * value


def show(ax, im, title="", **kw):
    ax.imshow(np.clip(im, 0, 1) if im.ndim == 3 else im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=1, **kw)
    ax.set_title(title, fontsize=10); ax.axis("off")


def result_figure(a, b, mask, result, path, titles=("A (where mask = 1)", "B (where mask = 0)")):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4 * a.shape[0] / a.shape[1] + 0.6))
    show(axes[0], a, titles[0]); show(axes[1], b, titles[1])
    show(axes[2], mask.squeeze(), "mask"); show(axes[3], result, "multiresolution blend")
    axes[2].axis("on"); axes[2].set_xticks([]); axes[2].set_yticks([])  # keep the frame: a white mask on a white page looks empty
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def color_vs_gray_figure(a, b, mask, path, sigmas=SIGMAS):
    ga, gb = gray(a), gray(b)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3 * a.shape[0] / a.shape[1] + 0.5))
    show(axes[0], blend(ga, gb, mask.squeeze(), sigmas), "grayscale blend")
    show(axes[1], blend(a, b, mask, sigmas), "colour blend (each channel blended separately)")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def method_comparison(a, b, mask, path):
    """Hard seam vs. single-scale feathering vs. multiresolution blending."""
    m2 = mask.squeeze()
    hard = mask * a + (1 - mask) * b
    feather = blur(m2, 16)[..., None] if a.ndim == 3 else blur(m2, 16)
    single = feather * a + (1 - feather) * b
    levels = [(n, blend(a, b, mask, SIGMAS[:n])) for n in (1, 2, 3, 4, 6)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.6 * a.shape[0] / a.shape[1] + 0.8))
    ims = [("hard seam (mask as is)", hard), ("single-scale feather, sigma 16", single)] + \
          [(f"multiresolution, {n} level{'s' if n > 1 else ''}", im) for n, im in levels]
    for ax, (t, im) in zip(axes.ravel(), ims):
        show(ax, im, t)
    axes.ravel()[-1].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def overlay_figure(a, b, mask, path):
    """Debug / documentation view: the mask outline drawn on both inputs."""
    m = (mask.squeeze() > 0.2).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6 * a.shape[0] / a.shape[1] + 0.4))
    for ax, im, t in ((axes[0], a, "A + mask outline"), (axes[1], b, "B + mask outline")):
        show(ax, im, t)
        for c in cs:
            ax.plot(c[:, 0, 0], c[:, 0, 1], "r-", lw=1.5)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def build_customs():
    """name -> (A, B, mask). Result = blend(A, B, mask): A shows where the mask is 1."""
    out = {}

    # 1. straight vertical seam: Da Qiao | Sun Ce, aligned on the eyes (same alignment as the hybrids)
    cfg = dict(ref=DATA / "daqiao.png", ref_pts=[(343, 149), (386, 138)],
               oth=DATA / "sunce.png", oth_pts=[(331, 153), (355, 164)],
               zoom=2.0, canvas=(600, 700), anchor=(300, 250))
    dq, sc = align(cfg)[2:4]
    m = np.zeros(dq.shape[:2]); m[:, :300] = 1
    out["daqiao_sunce_seam"] = (dq, sc, m[..., None])

    # 2. irregular mask: Cao Zhi's face inside Xiao Qiao's hair and background
    xq, cz = align(PAIRS["xiaoqiao_caozhi"])[2:4]
    out["face_swap"] = (cz, xq, ellipse_mask(xq.shape, FACE_CENTER, FACE_AXES, FACE_ANGLE))

    # 3. irregular, soft mask: the Taj Mahal seen in the lenses of the glasses
    selfie = load_rgb(DATA / "IMG_7415.jpg")
    taj = load_rgb(CARDS / "taj.jpg")
    pasted, mask = selfie.copy(), np.zeros(selfie.shape[:2])
    for (cx, cy), (ax_, ay_) in LENSES:
        x0, x1, y0, y1 = cx - ax_, cx + ax_, cy - ay_, cy + ay_
        crop = taj[20:20 + int(taj.shape[1] * 0 + 120), 50:50 + 200]            # the dome and arches
        pasted[y0:y1, x0:x1] = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        mask += ellipse_mask(selfie.shape[:2], (cx, cy), (ax_, ay_), value=LENS_ALPHA)[..., 0] if False else \
            ellipse_mask(selfie.shape[:2], (cx, cy), (ax_, ay_), value=LENS_ALPHA)
    out["taj_glasses"] = (pasted, selfie, np.clip(mask, 0, 1)[..., None])
    return out


# hand-placed shapes (pixels in the aligned canvases / the 1024 px selfie), checked on the overlay figures
FACE_CENTER, FACE_AXES, FACE_ANGLE = (245, 305), (108, 128), 0
LENSES = [((402, 525), (95, 55)), ((680, 530), (100, 58))]
LENS_ALPHA = 0.75


if __name__ == "__main__":
    apple, orange = load_rgb(APPLE), load_rgb(ORANGE)
    mask = step_mask(apple.shape)
    oraple = blend(apple, orange, mask)
    result_figure(apple, orange, mask, oraple, OUT / "part2_4_oraple.png", ("apple", "orange"))
    method_comparison(apple, orange, mask, OUT / "part2_4_oraple_methods.png")
    color_vs_gray_figure(apple, orange, mask, OUT / "part2_4_oraple_color.png")

    for name, (a, b, m) in build_customs().items():
        res = np.clip(blend(a, b, m), 0, 1)
        overlay_figure(a, b, m, OUT / f"part2_4_{name}_mask.png")
        result_figure(a, b, m, res, OUT / f"part2_4_{name}.png")
        process_figure(a, b, m, OUT / f"part2_4_{name}_process.png")
        print("done", name)
