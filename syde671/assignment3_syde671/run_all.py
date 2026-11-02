"""A.4 manual mosaic and Part B (automatic stitching) for IMG_4826 / IMG_4827 / IMG_4828.

    python a2.py && python a3.py && python run_all.py

IMG_4828 is the reference frame. Manual homographies come from points/*.json (see A2_notes.md).
Writes to outputs/: A4_*, B1_*, B2_*, B3_*, B4_*, compare_*.
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
REF = 2
N_ANMS, RATIO = 1500, 0.7


def load(n):
    return iio.imread(f'data/{n}.jpg')[:, :, :3] / 255.


def u8(im):
    return (np.clip(im, 0, 1) * 255).astype(np.uint8)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(f'{OUT}/{name}', dpi=110)
    plt.close()


ims = [load(n) for n in NAMES]

# ---------------- features for all three images
feat = []
for im in ims:
    _, p, s = harris(im)
    q, _ = anms(p, s, N_ANMS)
    feat.append((p, s, q, extract_descriptors(im, q)))

# ---------------- B.1 corners with / without ANMS (reference image)
p, s, q, _ = feat[REF]
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].imshow(ims[REF]); ax[0].plot(p[:, 0], p[:, 1], 'r.', ms=1.5); ax[0].set_title(f'Harris corners ({len(p)})')
ax[1].imshow(ims[REF]); ax[1].plot(q[:, 0], q[:, 1], 'r.', ms=4); ax[1].set_title(f'After ANMS ({len(q)})')
for x in ax: x.axis('off')
save_fig('B1_corners.png')

# ---------------- B.3 matching + RANSAC for each link to the reference
Hauto, lines, MI = {}, [], {}
for i in (0, 1):
    p1, p2 = feat[i][2], feat[REF][2]
    m = match_features(feat[i][3], feat[REF][3], RATIO)
    H, inl = ransac_homography(p1[m[:, 0]], p2[m[:, 1]])
    Hauto[i] = H
    MI[i] = (m, inl)
    lines.append(f'{NAMES[i]} -> {NAMES[REF]}: {len(m)} ratio-test matches, {inl.sum()} RANSAC inliers\nH =\n{np.round(H, 5)}\n')
    W = ims[i].shape[1]
    plt.figure(figsize=(16, 6)); plt.imshow(np.hstack([ims[i], ims[REF]]))
    for (a, b), good in zip(m, inl):
        plt.plot([p1[a, 0], p2[b, 0] + W], [p1[a, 1], p2[b, 1]], '-', lw=.7, color='lime' if good else 'red')
    plt.title(f'{NAMES[i]} -> {NAMES[REF]}: {len(m)} matches, {inl.sum()} RANSAC inliers (green); outliers red'); plt.axis('off')
    save_fig(f'B3_{NAMES[i]}.png')
open(f'{OUT}/B3_report.txt', 'w').write('\n'.join(lines))
print('\n'.join(lines))

# ---------------- B.2 descriptors: spread-out RANSAC-inlier features seen in two images
# Top row: 40x40 window in IMG_4827; second row: its 8x8 normalised descriptor; third row:
# the matching window in IMG_4828; bottom row: that image's descriptor.
m, inl = MI[1]
(q1, d1), (q2, d2) = (feat[1][2], feat[1][3]), (feat[REF][2], feat[REF][3])
good = m[inl]
pick = [0]
dist = np.linalg.norm(q1[good[:, 0]] - q1[good[pick[0], 0]], axis=1)
for _ in range(7):
    k = int(dist.argmax()); pick.append(k)
    dist = np.minimum(dist, np.linalg.norm(q1[good[:, 0]] - q1[good[k, 0]], axis=1))
g1, g2 = to_gray(ims[1]), to_gray(ims[REF])
fig, ax = plt.subplots(4, 8, figsize=(16, 8.4))
for c, k in enumerate(pick):
    i, j = good[k]
    for r, (g, q, d, tag) in enumerate(((g1, q1, d1, 'IMG_4827'), (g2, q2, d2, 'IMG_4828'))):
        x, y = (int(v) for v in q[i if r == 0 else j])
        ax[2 * r, c].imshow(g[y - 20:y + 20, x - 20:x + 20], cmap='gray'); ax[2 * r, c].set_title(f'{tag} 40x40', fontsize=8)
        ax[2 * r + 1, c].imshow(d[i if r == 0 else j].reshape(8, 8), cmap='gray'); ax[2 * r + 1, c].set_title('8x8 normalised', fontsize=8)
    dd = np.linalg.norm(d1[i] - d2[j])
    ax[3, c].set_xlabel(f'distance {dd:.2f}', fontsize=8)
for a_ in ax.ravel(): a_.set_xticks([]); a_.set_yticks([])
save_fig('B2_descriptors.png')

# numerical check of bias/gain invariance: I -> a*I + b leaves the descriptors unchanged
im = ims[REF]; d0 = extract_descriptors(im, feat[REF][2][:200])
d_t = extract_descriptors(0.6 * im + 0.15, feat[REF][2][:200])
chk = (f'bias/gain test (I -> 0.6 I + 0.15, 200 descriptors): max |difference| {np.abs(d0 - d_t).max():.2e}; '
       f'mean {d0.mean(1).max():.1e}, std {d0.std(1).min():.3f}..{d0.std(1).max():.3f}; '
       f'matched pairs shown: mean distance {np.mean([np.linalg.norm(d1[good[k][0]] - d2[good[k][1]]) for k in pick]):.2f} '
       f'vs random pairs {np.mean(np.linalg.norm(d1[:200] - d2[np.random.default_rng(0).permutation(len(d2))[:200]], axis=1)):.2f}')
open(f'{OUT}/B2_report.txt', 'w').write(chk)
print(chk)

# ---------------- manual homographies from A.2
Hman = {}
for i, f in ((0, 'p4826_4828'), (1, 'p4827_4828')):
    p1, p2 = (np.array(a, float) for a in json.load(open(f'points/{f}.json'))['points'])
    Hman[i] = computeH(p1, p2)

# ---------------- A.4 manual mosaic (2-band vs feather only); B.4 automatic mosaic
I = np.eye(3)
man, _, _ = blend(ims, [Hman[0], Hman[1], I])
fea, _, _ = blend(ims, [Hman[0], Hman[1], I], bands=1)
aut, _, _ = blend(ims, [Hauto[0], Hauto[1], I])
iio.imwrite(f'{OUT}/A4_manual.jpg', u8(man))
iio.imwrite(f'{OUT}/A4_feather_only.jpg', u8(fea))
iio.imwrite(f'{OUT}/B4_auto.jpg', u8(aut))

fig, ax = plt.subplots(2, 1, figsize=(14, 11))
ax[0].imshow(man); ax[0].set_title('manual correspondences (Part A)')
ax[1].imshow(aut); ax[1].set_title(f'automatic: Harris + ANMS({N_ANMS}) + MOPS + ratio {RATIO} + RANSAC (Part B)')
for x in ax: x.axis('off')
save_fig('compare_manual_vs_auto.png')
