import matplotlib
matplotlib.use('Agg')
import numpy as np
from p1 import *

np.set_printoptions(precision=6, suppress=False, linewidth=120)
out = open('../outputs/p1_results.txt', 'w')
for name in ['set1', 'set2']:
    d = 'data/' + name
    im1, im2 = imread(d + '/image1.jpg'), imread(d + '/image2.jpg')
    p1 = get_data_from_txt_file(d + '/pt_2D_1.txt')
    p2 = get_data_from_txt_file(d + '/pt_2D_2.txt')
    for tag, f in [('lls', lls_eight_point_alg), ('normalized', normalized_eight_point_alg)]:
        F = f(p1, p2)
        d1 = compute_distance_to_epipolar_lines(p1, p2, F)
        d2 = compute_distance_to_epipolar_lines(p2, p1, F.T)
        out.write(f'== {name} {tag}\nF =\n{np.array2string(F, precision=4, floatmode="fixed")}\n'
                  f'F (sci) =\n{np.array2string(F, formatter={"float": lambda x: f"{x:.4e}"})}\n'
                  f'rank={np.linalg.matrix_rank(F)}  dist_img1={d1:.4f}  dist_img2={d2:.4f}\n\n')
        plot_epipolar_lines_on_images(p1, p2, im1, im2, F)
        plt.savefig(f'../outputs/p1_{name}_{tag}.png', dpi=130, bbox_inches='tight')
        plt.close('all')
out.close()
print(open('../outputs/p1_results.txt').read())
