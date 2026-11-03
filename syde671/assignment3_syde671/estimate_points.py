"""Finds the matching point in image 2 for landmarks chosen by eye in image 1 (A.2 helper).

    python estimate_points.py river
    python estimate_points.py sea

The landmark positions in image 1 are read off gridded views (listed below). Each one is
located in image 2 by normalised cross-correlation of a 51x51 grey patch over the WHOLE
second image at several scales (the second photograph is a little zoomed). Scaling uses
warpImageBilinear from stitch.py. Nothing from the automatic Part B pipeline is used, so the
result can be compared with it independently. Writes points/p<a>_<b>.json and
outputs/check_<set>.png (red = image 1 point, cyan = found point, number = NCC).
"""
import json
import sys
import numpy as np
import imageio.v3 as iio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import match_template
from stitch import warpImageBilinear, to_gray

SETS = {
    'river': ('IMG_3361', 'IMG_3364', {
        'orange tree apex': (600, 142), 'left orange tree apex': (345, 162), 'right orange tree apex': (850, 160),
        'dark conifer top': (545, 205), 'yellow bush': (1095, 435), 'light bush': (525, 440),
        'shore notch': (650, 505), 'river stone left': (705, 548), 'river stone right': (950, 548),
        'big white stone': (240, 755), 'stone pair': (152, 722), 'dark rock': (372, 775),
        'rock lower': (437, 780), 'stone mid': (500, 710)}),
    'sea': ('IMG_4806', 'IMG_4807', {
        'left rock': (112, 466), 'dark flat rock': (228, 520), 'wave tip': (440, 452),
        'reef left': (770, 518), 'reef right': (960, 524), 'bright cloud': (480, 175),
        'cloud streak': (215, 152), 'small cloud': (620, 300), 'bush top left': (135, 548),
        'bush top mid': (525, 608), 'bush top right': (640, 588), 'flower bush': (1100, 545),
        'tree top': (255, 570)}),
}
# Accepted after looking at outputs/check_<set>.png: a candidate is kept when NCC >= 0.6, its
# runner-up (best peak > 30 px away) is clearly lower, and the pair is visibly the same feature.
# Elongated or smooth cloud features have a high runner-up on their own streak, so those two
# (cloud streak, small cloud) were accepted on visual agreement instead. Rejected candidates:
# sky-only patches (match anywhere: NCC 0.98 with runner-up 0.98), NCC < 0.6, and ambiguous
# repeats (runner-up within about 15% of the best peak) on rocks and reef that landed in the sky.
ACCEPT = {
    'river': ['orange tree apex', 'left orange tree apex', 'dark conifer top', 'yellow bush', 'light bush',
              'shore notch', 'river stone left', 'river stone right', 'big white stone', 'stone mid'],
    'sea': ['dark flat rock', 'reef right', 'bright cloud', 'cloud streak', 'small cloud', 'bush top mid',
            'bush top right', 'flower bush', 'tree top'],
}
HALF = 25
SCALES = np.arange(0.9, 1.45, 0.05)


def locate(g1, g2, x, y):
    patch = g1[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
    best = (-1, None, None, None)
    for s in SCALES:
        S = np.diag([s, s, 1.])
        n = int(round(s * (2 * HALF + 1)))
        t, _, _ = warpImageBilinear(patch, S, (0, 0, n - 1, n - 1))
        t = t[..., 0]
        if t.shape[0] > g2.shape[0] or t.shape[1] > g2.shape[1]:
            continue
        r = match_template(g2, t)
        iy, ix = np.unravel_index(r.argmax(), r.shape)
        if r[iy, ix] > best[0]:
            yy, xx = np.mgrid[:r.shape[0], :r.shape[1]]
            far = r[(yy - iy) ** 2 + (xx - ix) ** 2 > 30 ** 2]
            best = (float(r[iy, ix]), (ix + n // 2, iy + n // 2), float(s), float(far.max()))
    return best


if __name__ == '__main__':
    name = sys.argv[1]
    a, b, marks = SETS[name]
    im1, im2 = (iio.imread(f'data/{n}.jpg')[:, :, :3] / 255. for n in (a, b))
    g1, g2 = to_gray(im1), to_gray(im2)
    rows = []
    for k, (x, y) in marks.items():
        ncc, p2, s, second = locate(g1, g2, x, y)
        rows.append((k, (x, y), p2, ncc, s, second))
        print(f'{k:24s} ({x:4d},{y:4d}) -> {p2}  ncc {ncc:.2f}  scale {s:.2f}  runner-up {second:.2f}')
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.2))
    for x_, im in zip(ax, (im1, im2)): x_.imshow(im); x_.axis('off')
    for i, (k, p1, p2, ncc, s, second) in enumerate(rows):
        ax[0].plot(*p1, 'r+', ms=12, mew=2); ax[0].text(p1[0] + 8, p1[1] - 8, str(i + 1), color='yellow', fontsize=11)
        ax[1].plot(*p2, 'c+', ms=12, mew=2); ax[1].text(p2[0] + 8, p2[1] - 8, f'{i + 1}:{ncc:.2f}', color='yellow', fontsize=10)
    plt.tight_layout(); plt.savefig(f'outputs/check_{name}.png', dpi=110); plt.close()
    keep = [r for r in rows if r[0] in ACCEPT[name]]
    json.dump({'images': [f'data/{a}.jpg', f'data/{b}.jpg'], 'names': [r[0] for r in keep],
               'points': [[list(map(float, r[1])) for r in keep], [list(map(float, r[2])) for r in keep]],
               'ncc': [r[3] for r in keep]},
              open(f'points/p{a[4:]}_{b[4:]}.json', 'w'), indent=1)
    print(f'accepted {len(keep)} of {len(rows)} -> points/p{a[4:]}_{b[4:]}.json')
    json.dump({'images': [f'data/{a}.jpg', f'data/{b}.jpg'], 'names': [r[0] for r in rows],
               'points': [[list(map(float, r[1])) for r in rows], [list(map(float, r[2])) for r in rows]],
               'ncc': [r[3] for r in rows], 'scale': [r[4] for r in rows], 'runner_up': [r[5] for r in rows]},
              open(f'points/candidates_{name}.json', 'w'), indent=1)
