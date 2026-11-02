"""A.3: rectification with inverse warping, nearest neighbour vs bilinear.

    python a3.py

Writes to outputs/: A3_<target>.png (source + quad, NN, bilinear), A3_<target>_zoom.png
(nearest vs bilinear crop at 4x), A3_<target>_full.png (full warped canvas + alpha mask)
and A3_report.txt (timings and synthetic accuracy test).
"""
import time
import numpy as np
import imageio.v3 as iio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stitch import computeH, warpImageNearestNeighbor, warpImageBilinear, warp_bbox, apply_H

OUT = 'outputs'
# image, quad corners clicked clockwise from top-left (x, y), rectified size (w, h)
TARGETS = {
    'tower':          ('IMG_5185', [[184, 2], [336, 2], [334, 401], [130, 408]], (300, 600)),
    'banner_rotman':  ('IMG_1456', [[706, 171], [894, 105], [896, 528], [706, 551]], (300, 700)),
    'banner_roberta': ('IMG_1456', [[520, 245], [645, 198], [645, 557], [522, 574]], (300, 700)),
}


def load(n):
    return iio.imread(f'data/{n}.jpg')[:, :, :3] / 255.


def u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def timeit(f, reps=5):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)


lines = []
for name, (img, quad, (w, h)) in TARGETS.items():
    im = load(img)
    src = np.array(quad, float)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    H = computeH(src, dst)
    box = (0, 0, w, h)
    nn, a_nn, _ = warpImageNearestNeighbor(im, H, box)
    bl, a_bl, _ = warpImageBilinear(im, H, box)

    # ---- side by side
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.5))
    ax[0].imshow(im); q = np.vstack([src, src[:1]]); ax[0].plot(q[:, 0], q[:, 1], 'r-', lw=1.5)
    ax[0].set_title(f'{img}: quad'); ax[1].imshow(nn); ax[1].set_title('nearest neighbour')
    ax[2].imshow(bl); ax[2].set_title('bilinear')
    for x in ax: x.axis('off')
    plt.tight_layout(); plt.savefig(f'{OUT}/A3_{name}.png', dpi=110); plt.close()

    # ---- 4x zoom crop (render at 4x so the interpolation difference is visible)
    S = np.diag([4, 4, 1.]); H4 = S @ H; big = (0, 0, 4 * w, 4 * h)
    nn4, _, _ = warpImageNearestNeighbor(im, H4, big); bl4, _, _ = warpImageBilinear(im, H4, big)
    cy, cx = int(4 * h * .35), int(4 * w * .45)
    sl = (slice(cy, cy + 160), slice(cx, cx + 160))
    iio.imwrite(f'{OUT}/A3_{name}_zoom.png', u8(np.hstack([nn4[sl], np.ones((160, 8, 3)), bl4[sl]])))

    # ---- full canvas with alpha (bounding box from the 4 transformed corners)
    bx = warp_bbox(im.shape, H)
    full = f'{bx[2] - bx[0] + 1}x{bx[3] - bx[1] + 1}'
    if (bx[2] - bx[0]) * (bx[3] - bx[1]) < 6e6:
        fb, fa, off = warpImageBilinear(im, H)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5)); ax[0].imshow(fb); ax[0].set_title(f'full bilinear canvas {full}')
        ax[1].imshow(fa, cmap='gray'); ax[1].set_title('alpha (1 = source pixel, 0 = empty)')
        for x in ax: x.axis('off')
        plt.tight_layout(); plt.savefig(f'{OUT}/A3_{name}_full.png', dpi=100); plt.close()
    else:
        full += ' (too large to plot)'

    # ---- timing at three output scales (best of 5)
    tm = {}
    for s in (1, 2, 4):
        Hs = np.diag([s, s, 1.]) @ H; bs = (0, 0, s * w, s * h)
        tm[s] = (timeit(lambda: warpImageNearestNeighbor(im, Hs, bs)), timeit(lambda: warpImageBilinear(im, Hs, bs)))

    lines.append(f'== {name} ({img}) -> {w}x{h}\nH =\n{np.round(H, 5)}\nfull-canvas size from corners: {full}')
    for s, (a, b) in tm.items():
        lines.append(f'  {s * w}x{s * h}: nearest {a * 1e3:6.1f} ms  bilinear {b * 1e3:6.1f} ms  ratio {b / a:.2f}x')
    lines.append('')

# ---- accuracy test with a known answer: the plane texture is an analytic function f(u, v);
# the "photo" is f sampled exactly through a perspective view (no interpolation anywhere), then
# rectified back with each method and compared to f. Uses the tower homography.
name, (img, quad, (w, h)) = 'tower', TARGETS['tower']
H = computeH(np.array(quad, float), np.array([[0, 0], [w, 0], [w, h], [0, h]], float))  # photo -> plane
f = lambda u, v: 0.5 + 0.2 * np.sin(2 * np.pi * u / 30) + 0.2 * np.sin(2 * np.pi * v / 37) + 0.1 * np.sin(2 * np.pi * (u + v) / 23)
ph, pw = 900, 1200
xs, ys = np.meshgrid(np.arange(pw), np.arange(ph))
p = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)]); q = H @ p
photo = f(q[0] / q[2], q[1] / q[2]).reshape(ph, pw)
photo3 = np.repeat(photo[..., None], 3, axis=2)
uu, vv = np.meshgrid(np.arange(w), np.arange(h)); truth = f(uu, vv)
lines.append('== synthetic accuracy test (tower homography, analytic texture, photo sampled exactly)')
for nm, fn in (('nearest', warpImageNearestNeighbor), ('bilinear', warpImageBilinear)):
    r, a, _ = fn(photo3, H, (0, 0, w - 1, h - 1))
    m = a > 0
    err = r[..., 0][m] - truth[m]
    lines.append(f'  {nm:9s} RMSE {np.sqrt(np.mean(err ** 2)):.4f}  PSNR {10 * np.log10(1 / np.mean(err ** 2)):.2f} dB  (valid px {m.sum()})')

open(f'{OUT}/A3_report.txt', 'w').write('\n'.join(lines))
print('\n'.join(lines))
