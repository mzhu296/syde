import matplotlib
matplotlib.use('Agg')
import numpy as np
from p2 import *

np.set_printoptions(precision=4, suppress=False, linewidth=120)
res = open('../outputs/p2_results.txt', 'w')
for name in ['set1', 'set2']:
    d = 'data/' + name
    # set2 images are stored as 3-channel; the rectifier expects grayscale
    im1, im2 = [(lambda a: a if a.ndim == 2 else a.mean(axis=2))(imread(d + f'/image{i}.jpg')) for i in (1, 2)]
    p1 = get_data_from_txt_file(d + '/pt_2D_1.txt')
    p2 = get_data_from_txt_file(d + '/pt_2D_2.txt')
    F = normalized_eight_point_alg(p1, p2)
    e1, e2 = compute_epipole(F.T), compute_epipole(F)
    H1, H2 = compute_matching_homographies(e2, F, im2, p1, p2)
    n1 = H1 @ p1.T; n2 = H2 @ p2.T
    n1 = (n1 / n1[2]).T; n2 = (n2 / n2[2]).T
    r1, o1 = compute_rectified_image(im1, H1)
    r2, o2 = compute_rectified_image(im2, H2)
    n1 -= o1 + (0,); n2 -= o2 + (0,)
    # same-row check in the original (un-offset) rectified frame
    dy = np.abs((H1 @ p1.T)[1] / (H1 @ p1.T)[2] - (H2 @ p2.T)[1] / (H2 @ p2.T)[2])
    res.write(f'== {name}\ne1={e1}\ne2={e2}\nH1=\n{H1}\nH2=\n{H2}\n'
              f'H2@e2={H2 @ e2}\nmean |dy| = {dy.mean():.4f} px, max = {dy.max():.4f} px\n\n')
    Fn = normalized_eight_point_alg(n1, n2)
    plot_epipolar_lines_on_images(n1, n2, r1, r2, Fn)
    plt.savefig(f'../outputs/p2_{name}_rectified_lines.png', dpi=130, bbox_inches='tight')
    plt.close('all')
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a_, im, pts, t in [(ax[0], r1, n1, 'Rectified image 1'), (ax[1], r2, n2, 'Rectified image 2')]:
        a_.imshow(im, cmap='gray'); a_.set_title(t); a_.axis('off')
        for y in pts[::4, 1]: a_.axhline(y, color='r', lw=0.6)
    plt.savefig(f'../outputs/p2_{name}_rectified_rows.png', dpi=130, bbox_inches='tight')
    plt.close('all')
res.close()
print(open('../outputs/p2_results.txt').read())
