import matplotlib
matplotlib.use('Agg')
import numpy as np
from p3 import *

res = open('../outputs/p3_results.txt', 'w')
for name in ['set1', 'set1_subset']:
    d = 'data/' + name
    p1 = get_data_from_txt_file(d + '/pt_2D_1.txt')
    p2 = get_data_from_txt_file(d + '/pt_2D_2.txt')
    gt = get_data_from_txt_file(d + '/pt_3D.txt')
    structure, motion = factorization_method(p1, p2)
    D = np.vstack([p1[:, :2].T, p2[:, :2].T]); D = D - D.mean(axis=1, keepdims=True)
    S = np.linalg.svd(D, compute_uv=False)
    # best affine map from recovered structure to ground truth (residual shows how far from affine-equivalent)
    X = np.vstack([structure, np.ones(structure.shape[1])]).T
    A, *_ = np.linalg.lstsq(X, gt, rcond=None)
    err = np.linalg.norm(X @ A - gt, axis=1)
    res.write(f'== {name}  N={p1.shape[0]}\nsingular values = {np.array2string(S, precision=4)}\n'
              f'rmse of best affine fit to ground truth = {np.sqrt((err**2).mean()):.4f} '
              f'(GT extent {np.ptp(gt, axis=0).round(1)})\n\n')
    for ang, tag in [((20, -60), 'a'), ((20, 30), 'b')]:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        scatter_3D_axis_equal(structure[0], structure[1], structure[2], ax)
        ax.set_title('Factorization Method'); ax.view_init(*ang)
        ax = fig.add_subplot(122, projection='3d')
        scatter_3D_axis_equal(gt[:, 0], gt[:, 1], gt[:, 2], ax)
        ax.set_title('Ground Truth'); ax.view_init(*ang)
        plt.savefig(f'../outputs/p3_{name}_{tag}.png', dpi=130, bbox_inches='tight'); plt.close('all')
res.close()
print(open('../outputs/p3_results.txt').read())
