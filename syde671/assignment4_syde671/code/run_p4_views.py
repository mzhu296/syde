import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

S = np.load('../outputs/p4_dense_structure.npy')
x, y, z = S.T
# clip to the same box the starter plot uses so outliers don't stretch the axes
m = (abs(x) < 5) & (abs(y) < 5) & (z > 0) & (z < 10)
x, y, z = x[m], y[m], z[m]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# y points down in image coordinates, so invert it to get an upright statue
for a, (u, v, t, xl, yl) in zip(ax, [
        (x, y, 'Front view (camera)', 'x', 'y'),
        (z, y, 'Side view', 'depth z', 'y'),
        (x, z, 'Top view', 'x', 'depth z')]):
    a.scatter(u, v, c='k', s=0.5, alpha=0.5)
    a.set_title(t); a.set_xlabel(xl); a.set_ylabel(yl); a.set_aspect('equal')
    if yl == 'y': a.invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/p4_statue_views.png', dpi=130)
print(m.sum(), 'of', len(m), 'points inside the plotted box;', 'z range', z.min().round(2), z.max().round(2))
