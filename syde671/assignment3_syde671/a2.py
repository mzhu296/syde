"""A.2: recover homographies from the point files in points/.

    python a2.py

For each pair writes outputs/A2_<pair>.png (numbered correspondences) and
outputs/A2_<pair>.txt (points, the 2n x 8 system A h = b, recovered H, residuals).
"""
import json
import numpy as np
import imageio.v3 as iio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stitch import computeH, apply_H

PAIRS = ['p4827_4828', 'p4826_4828', 'p3361_3364', 'p4806_4807']


def system(p1, p2):
    """Rows of A h = b with h = (a, b, c, d, e, f, g, h), h33 = 1."""
    A = np.zeros((2 * len(p1), 8))
    b = p2.reshape(-1)
    for i, ((x, y), (u, v)) in enumerate(zip(p1, p2)):
        A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
    return A, b


for name in PAIRS:
    d = json.load(open(f'points/{name}.json'))
    p1, p2 = (np.array(p, float) for p in d['points'])
    H = computeH(p1, p2)
    A, b = system(p1, p2)
    h = H.ravel()[:8]
    err = np.linalg.norm(apply_H(H, p1) - p2, axis=1)
    with open(f'outputs/A2_{name}.txt', 'w') as f, np.printoptions(precision=2, suppress=True, linewidth=220):
        f.write(f'{name}: {d["images"][0]} -> {d["images"][1]}, n = {len(p1)} correspondences\n\n')
        for i, (nm, a, c, e) in enumerate(zip(d['names'], p1, p2, err)):
            f.write(f'{i + 1:2d} {nm:18s} im1 ({a[0]:7.1f},{a[1]:7.1f})  im2 ({c[0]:7.1f},{c[1]:7.1f})  residual {e:.2f} px\n')
        f.write(f'\nSystem A h = b  (A is {A.shape[0]}x8, solved by least squares)\nA =\n{A}\nb =\n{b}\n')
        f.write(f'\nh = {h}\n\nH =\n{H}\n\nresidual mean {err.mean():.2f} px, max {err.max():.2f} px\n')
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.2))
    for x, im, c in zip(ax, d['images'], (p1, p2)):
        x.imshow(iio.imread(im)); x.plot(c[:, 0], c[:, 1], 'r+', ms=14, mew=2)
        for i, (px, py) in enumerate(c):
            x.text(px + 8, py - 8, str(i + 1), color='yellow', fontsize=13, weight='bold')
        x.set_title(im.split('/')[-1]); x.axis('off')
    plt.tight_layout(); plt.savefig(f'outputs/A2_{name}.png', dpi=110); plt.close()
    print(name, 'n =', len(p1), 'residual mean %.2f max %.2f' % (err.mean(), err.max()))
    print(np.round(H, 4))
