"""Extra figures and numbers for the report (run after a2.py, a3.py, run_all.py).

    python report_figs.py

Writes outputs/A1_sources.png, outputs/A4_blend_compare.png and prints how far the manual and
automatic homographies disagree at the image corners.
"""
import json
import numpy as np
import imageio.v3 as iio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stitch import *

OUT = 'outputs'
NAMES = ['IMG_4826', 'IMG_4827', 'IMG_4828']
ims = [iio.imread(f'data/{n}.jpg')[:, :, :3] / 255. for n in NAMES]

# ---- A.1 source images
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, im, n in zip(ax, ims, NAMES):
    a.imshow(im); a.set_title(f'{n}  ({im.shape[1]}x{im.shape[0]})'); a.axis('off')
plt.tight_layout(); plt.savefig(f'{OUT}/A1_sources.png', dpi=110); plt.close()

# ---- A.4 feather-only vs 2-band, crop at the 4827/4828 overlap and the cliff
a = iio.imread(f'{OUT}/A4_feather_only.jpg'); b = iio.imread(f'{OUT}/A4_manual.jpg')
h, w = a.shape[:2]
sl = (slice(int(h * .42), int(h * .70)), slice(int(w * .52), int(w * .78)))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
ax[0].imshow(a[sl]); ax[0].set_title('feather only (1 band)')
ax[1].imshow(b[sl]); ax[1].set_title('2-band blend (sigma 15 low pass, sigma 2.5 high-frequency hand-over)')
for x in ax: x.axis('off')
plt.tight_layout(); plt.savefig(f'{OUT}/A4_blend_compare.png', dpi=110); plt.close()

# ---- manual vs automatic H: where do the four image corners land in the reference frame?
auto = {}
feat = []
for im in ims:
    _, p, s = harris(im); q, _ = anms(p, s, 1500); feat.append((q, extract_descriptors(im, q)))
for i, f in ((0, 'p4826_4828'), (1, 'p4827_4828')):
    m = match_features(feat[i][1], feat[2][1], 0.7)
    Ha, inl = ransac_homography(feat[i][0][m[:, 0]], feat[2][0][m[:, 1]])
    p1, p2 = (np.array(x, float) for x in json.load(open(f'points/{f}.json'))['points'])
    Hm = computeH(p1, p2)
    H_, W_ = ims[i].shape[:2]
    corners = np.array([[0, 0], [W_, 0], [W_, H_], [0, H_]], float)
    d = np.linalg.norm(apply_H(Hm, corners) - apply_H(Ha, corners), axis=1)
    pts = np.array([[W_ / 2, H_ / 2]])
    dc = np.linalg.norm(apply_H(Hm, pts) - apply_H(Ha, pts))
    print(f'{NAMES[i]}: manual vs automatic corner disagreement (px) {np.round(d, 1)}, centre {dc:.1f}')
