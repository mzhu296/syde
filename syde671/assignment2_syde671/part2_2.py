"""Part 2.2: hybrid images (Oliva, Torralba & Schyns 2006)."""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from part1_2 import HERE, OUT
from part2_1 import load_rgb

HYB = HERE / "hybrid"
DATA = HERE / "data"

# Two hand-picked reference points (the eyes) per image, as (x, y) in pixels. ref = stays unrotated.
PAIRS = {
    "derek_nutmeg": dict(
        ref=HYB / "DerekPicture.jpg", ref_pts=[(300, 345), (437, 331)],
        oth=HYB / "nutmeg.jpg", oth_pts=[(606, 289), (750, 368)],
        zoom=1.3, canvas=(560, 600), anchor=(280, 260),
    ),
    # card art is low resolution (faces ~50 px wide), so it is enlarged ~3.5x to a similar eye distance as Derek
    "xiaoqiao_caozhi": dict(
        ref=DATA / "xiaoqiao.png", ref_pts=[(262, 148), (293, 151)],
        oth=DATA / "caozhi.png", oth_pts=[(296, 177), (332, 186)],
        zoom=3.8, canvas=(480, 520), anchor=(240, 220),
    ),
    "zhaoyun_zhaotong": dict(
        ref=DATA / "zhaoyun.png", ref_pts=[(262, 202), (297, 204)],
        oth=DATA / "zhaotong.png", oth_pts=[(126, 326), (168, 321)],
        zoom=3.4, canvas=(480, 520), anchor=(240, 220),
    ),
}


# Which image supplies the low frequencies, and the Gaussian widths (in pixels of the aligned canvas).
HYBRIDS = [
    dict(name="derek_nutmeg", pair="derek_nutmeg", low="ref", sigma_low=10, sigma_high=5),
    # the crisp, line-rich painting supplies the high frequencies; the soft, colour-rich one the low frequencies
    dict(name="xiaoqiao_caozhi", pair="xiaoqiao_caozhi", low="oth", sigma_low=8, sigma_high=4),   # Cao Zhi far, Xiao Qiao near
    dict(name="zhaoyun_zhaotong", pair="zhaoyun_zhaotong", low="oth", sigma_low=8, sigma_high=4),  # Zhao Tong far, Zhao Yun near
]


def similarity(p, q):
    """2x3 matrix of the scale + rotation + translation that maps the two points p onto the two points q.

    Treat points as complex numbers: q = a * p + t, where |a| is the scale and arg(a) the rotation.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    a = complex(*(q[1] - q[0])) / complex(*(p[1] - p[0]))
    t = complex(*q[0]) - a * complex(*p[0])
    return np.array([[a.real, -a.imag, t.real], [a.imag, a.real, t.imag]])


def align(cfg):
    """Warp both images onto a common canvas so that their two eye points coincide."""
    ref, oth = load_rgb(cfg["ref"]), load_rgb(cfg["oth"])
    r = np.asarray(cfg["ref_pts"], float)
    mid = r.mean(axis=0)
    target = cfg["zoom"] * (r - mid) + np.asarray(cfg["anchor"], float)  # reference eyes, zoomed, on the canvas

    def warp(img, pts):
        M = similarity(pts, target)
        return np.clip(cv2.warpAffine(img.astype(np.float32), M, cfg["canvas"], flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REFLECT), 0, 1).astype(float)

    return ref, oth, warp(ref, cfg["ref_pts"]), warp(oth, cfg["oth_pts"]), target


def gray(img):
    return img if img.ndim == 2 else img @ np.array([0.299, 0.587, 0.114])


def lowpass(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def hybrid(low_img, high_img, s_low, s_high):
    """low-pass of one image + high-pass (image minus its own Gaussian blur) of the other."""
    low = lowpass(low_img, s_low)
    high = high_img - lowpass(high_img, s_high)
    return np.clip(low + high, 0, 1), low, high


def spectrum(g):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))) + 1e-8)


def show(ax, im, title="", **kw):
    ax.imshow(np.clip(im, 0, 1) if kw.pop("clip", True) else im, cmap="gray" if im.ndim == 2 else None,
              vmin=kw.pop("vmin", 0), vmax=kw.pop("vmax", 1), **kw)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def distance_strip(ax_row, img):
    for ax, f in zip(ax_row, [1, 0.5, 0.25, 0.125]):
        small = cv2.resize(img.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        show(ax, small, f"{int(f * 100)}% size")


def run_pair(h):
    name, cfg = h["name"], dict(PAIRS[h["pair"]], **h)
    ref, oth, a_ref, a_oth, target = align(cfg)
    low_is_ref = cfg["low"] == "ref"
    low_img, high_img = (a_ref, a_oth) if low_is_ref else (a_oth, a_ref)
    hyb, low, high = hybrid(low_img, high_img, cfg["sigma_low"], cfg["sigma_high"])

    # alignment figure: originals with the chosen points, then the aligned pair
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6))
    for ax, img, pts, title in [(axes[0], ref, cfg["ref_pts"], "reference image (keeps its orientation)"),
                                (axes[1], oth, cfg["oth_pts"], "second image (will be rotated and scaled)")]:
        show(ax, img, title)
        ax.plot(*zip(*pts), "r+", ms=14, mew=2)
    show(axes[2], a_ref, "reference, aligned"); show(axes[3], a_oth, "second image, aligned")
    for ax in axes[2:]:
        ax.plot(*zip(*target), "r+", ms=14, mew=2)
    fig.tight_layout(); fig.savefig(OUT / f"part2_2_{name}_align.png", dpi=100); plt.close(fig)

    # final hybrid at several viewing distances
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=[1, 0.5, 0.25, 0.125]))
    distance_strip(axes, hyb)
    fig.tight_layout(); fig.savefig(OUT / f"part2_2_{name}_distance.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6.4)); show(ax, hyb, f"hybrid: sigma_low={cfg['sigma_low']}, sigma_high={cfg['sigma_high']}")
    fig.tight_layout(); fig.savefig(OUT / f"part2_2_{name}_hybrid.png", dpi=100); plt.close(fig)
    return cfg, low_img, high_img, hyb, low, high


def cutoff(sigma):
    """Standard deviation of the Gaussian's frequency response, in cycles per pixel."""
    return 1 / (2 * np.pi * sigma)


def fourier_figure(name, low_img, high_img, hyb, low, high):
    """Spatial row and log-magnitude-spectrum row: inputs, filtered images, hybrid (grayscale)."""
    gl, gh, glow, ghigh, ghyb = map(gray, (low_img, high_img, low, high, hyb))
    cols = [(gl, "low-frequency source"), (gh, "high-frequency source"),
            (glow, "low-pass filtered"), (0.5 + ghigh, "high-pass filtered (+0.5)"), (ghyb, "hybrid")]
    specs = [gl, gh, glow, ghigh, ghyb]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.4))
    for c, ((im, title), g) in enumerate(zip(cols, specs)):
        show(axes[0, c], im, title)
        axes[1, c].imshow(spectrum(g), cmap="viridis", vmin=-4, vmax=10)
        axes[1, c].set_title("log |FFT|", fontsize=10); axes[1, c].axis("off")
    fig.tight_layout(); fig.savefig(OUT / f"part2_2_{name}_fourier.png", dpi=90); plt.close(fig)


def cutoff_sweep(name, low_img, high_img, sigmas_low, sigmas_high):
    gl, gh = gray(low_img), gray(high_img)
    fig, axes = plt.subplots(len(sigmas_high), len(sigmas_low), figsize=(4.2 * len(sigmas_low), 4.5 * len(sigmas_high)))
    for i, sh in enumerate(sigmas_high):
        for j, sl in enumerate(sigmas_low):
            h, _, _ = hybrid(gl, gh, sl, sh)
            show(axes[i, j], h, f"sigma_low={sl} (f={cutoff(sl):.3f}), sigma_high={sh} (f={cutoff(sh):.3f})")
    fig.tight_layout(); fig.savefig(OUT / f"part2_2_{name}_cutoff_sweep.png", dpi=80); plt.close(fig)


def color_study(name, low_img, high_img, s_low, s_high):
    """Grayscale vs colour on the low part only, the high part only, or both."""
    gl, gh = gray(low_img), gray(high_img)
    to3 = lambda g: np.dstack([g] * 3)
    variants = [("grayscale", gl, gh), ("colour: low only", low_img, to3(gh)),
                ("colour: high only", to3(gl), high_img), ("colour: both", low_img, high_img)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.4))
    for c, (title, lo, hi) in enumerate(variants):
        h, _, _ = hybrid(lo, hi, s_low, s_high)
        show(axes[0, c], h, title)
        small = cv2.resize(h.astype(np.float32), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        show(axes[1, c], small, f"{title}: 25% size")
    fig.tight_layout(); fig.savefig(OUT / f"part2_2_{name}_color.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    for h in HYBRIDS:
        cfg, low_img, high_img, hyb, low, high = run_pair(h)
        print(f"{h['name']}: sigma_low={cfg['sigma_low']} (f={cutoff(cfg['sigma_low']):.4f} cyc/px), "
              f"sigma_high={cfg['sigma_high']} (f={cutoff(cfg['sigma_high']):.4f} cyc/px)")
        cutoff_sweep(h["name"], low_img, high_img, [4, 8, 12], [2, 4, 6] if h["name"] != "derek_nutmeg" else [3, 5, 8])
        if h["name"] == "derek_nutmeg":  # the example shown in full
            fourier_figure(h["name"], low_img, high_img, hyb, low, high)
            color_study(h["name"], low_img, high_img, cfg["sigma_low"], cfg["sigma_high"])
