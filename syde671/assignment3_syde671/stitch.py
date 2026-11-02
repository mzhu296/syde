import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, map_coordinates
from harris import get_harris_corners, dist2

# Convention: points are (x, y); images are float arrays in [0, 1], H x W x 3.
# H maps im1 -> im2 coordinates: p2 ~ H p1.


# ---------------------------------------------------------------- Part A

def computeH(im1_pts, im2_pts):
    """Least-squares homography with h33 = 1. Needs n >= 4 points."""
    p1 = np.asarray(im1_pts, float)
    p2 = np.asarray(im2_pts, float)
    n = len(p1)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    x, y = p1[:, 0], p1[:, 1]
    u, v = p2[:, 0], p2[:, 1]
    A[0::2, 0:3] = np.c_[x, y, np.ones(n)]
    A[0::2, 6] = -u * x
    A[0::2, 7] = -u * y
    A[1::2, 3:6] = np.c_[x, y, np.ones(n)]
    A[1::2, 6] = -v * x
    A[1::2, 7] = -v * y
    b[0::2] = u
    b[1::2] = v
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.append(h, 1.0).reshape(3, 3)


def apply_H(H, pts):
    pts = np.asarray(pts, float)
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def warp_bbox(shape, H):
    """Integer bounding box (xmin, ymin, xmax, ymax) of the warped image."""
    h, w = shape[:2]
    c = apply_H(H, [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return (int(np.floor(c[:, 0].min())), int(np.floor(c[:, 1].min())),
            int(np.ceil(c[:, 0].max())), int(np.ceil(c[:, 1].max())))


def _inverse_coords(shape, H, bbox):
    """Source (x, y) for each pixel of the output canvas defined by bbox."""
    xmin, ymin, xmax, ymax = bbox
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    q = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    s = np.linalg.inv(H) @ q
    sx, sy = s[0] / s[2], s[1] / s[2]
    return sx.reshape(xs.shape), sy.reshape(xs.shape)


def warpImageNearestNeighbor(im, H, bbox=None):
    """Inverse warp, nearest neighbour. Returns (warped, alpha, (xmin, ymin))."""
    h, w = im.shape[:2]
    bbox = bbox or warp_bbox(im.shape, H)
    sx, sy = _inverse_coords(im.shape, H, bbox)
    xi, yi = np.rint(sx).astype(int), np.rint(sy).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(sx.shape + im.shape[2:])
    out[valid] = im[yi[valid], xi[valid]]
    return out, valid.astype(float), (bbox[0], bbox[1])


def warpImageBilinear(im, H, bbox=None):
    """Inverse warp, bilinear (pixel centres at integer coords)."""
    h, w = im.shape[:2]
    bbox = bbox or warp_bbox(im.shape, H)
    sx, sy = _inverse_coords(im.shape, H, bbox)
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    x0 = np.clip(np.floor(sx).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(sy).astype(int), 0, h - 2)
    dx = np.clip(sx - x0, 0, 1)[..., None]
    dy = np.clip(sy - y0, 0, 1)[..., None]
    if im.ndim == 2:
        im = im[..., None]
    out = ((1 - dx) * (1 - dy) * im[y0, x0] + dx * (1 - dy) * im[y0, x0 + 1]
           + (1 - dx) * dy * im[y0 + 1, x0] + dx * dy * im[y0 + 1, x0 + 1])
    out[~valid] = 0
    return out, valid.astype(float), (bbox[0], bbox[1])


def feather_alpha(mask):
    """Weight that is 1 deep inside the image and falls to 0 at its border."""
    d = distance_transform_edt(mask > 0)
    return d / d.max() if d.max() > 0 else d


def blend(images, Hs, bands=2, sigma=15, sigma_hi=2.5, warp=warpImageBilinear):
    """Warp every image with its H into a shared canvas and blend.

    bands=1: feathered weighted average. bands=2: low frequencies are
    feathered, high frequencies come from the image with the largest weight
    (2-level Laplacian pyramid, hand-over softened by sigma_hi), which removes ghosting.
    """
    boxes = [warp_bbox(im.shape, H) for im, H in zip(images, Hs)]
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
           max(b[2] for b in boxes), max(b[3] for b in boxes))
    layers, weights = [], []
    for im, H in zip(images, Hs):
        wim, mask, _ = warp(im, H, box)
        layers.append(wim)
        weights.append(feather_alpha(mask) * (mask > 0))
    W = np.stack(weights)
    wsum = W.sum(0)
    wnorm = W / np.maximum(wsum, 1e-8)
    if bands == 1:
        out = sum(l * wn[..., None] for l, wn in zip(layers, wnorm))
    else:
        lows = [gaussian_filter(l, (sigma, sigma, 0)) for l in layers]
        highs = [l - lo for l, lo in zip(layers, lows)]
        low = sum(lo * wn[..., None] for lo, wn in zip(lows, wnorm))
        winner = W.argmax(0)
        # soft selection: blur the winner masks (sigma_hi px) so the high-frequency
        # hand-over is a narrow feather, not a hard line
        sel = np.stack([gaussian_filter((winner == i).astype(float), sigma_hi) for i in range(len(highs))])
        sel /= np.maximum(sel.sum(0), 1e-8)
        high = sum(hi * s[..., None] for hi, s in zip(highs, sel))
        out = low + high
    out[wsum == 0] = 0
    return np.clip(out, 0, 1), (wsum > 0), box[:2]


# ---------------------------------------------------------------- Part B

def to_gray(im):
    return im @ np.array([0.299, 0.587, 0.114]) if im.ndim == 3 else im


def harris(im, edge_discard=20):
    """Returns (h, pts) with pts as n x 2 (x, y) and strengths of each."""
    h, coords = get_harris_corners(to_gray(im), edge_discard)
    ys, xs = coords
    return h, np.c_[xs, ys], h[ys, xs]


def anms(pts, strengths, n_keep=500, c_robust=0.9, max_candidates=4000):
    """Adaptive non-maximal suppression (Brown et al., Sec. 3).

    The suppression radius of a point is its distance to the nearest point
    that is sufficiently stronger (f_i < c_robust * f_j). Keep the n_keep
    points with the largest radii. Only the max_candidates strongest corners
    are considered so the N x N distance matrix stays small.
    """
    top = np.argsort(-strengths)[:max_candidates]
    pts, strengths = pts[top], strengths[top]
    d2 = dist2(pts.astype(float), pts.astype(float))
    stronger = strengths[:, None] < c_robust * strengths[None, :]
    d2 = np.where(stronger, d2, np.inf)
    radii = d2.min(axis=1)
    keep = np.argsort(-radii)[:n_keep]
    return pts[keep], radii[keep]


def extract_descriptors(im, pts, window=40, size=8):
    """Axis-aligned size x size descriptors sampled (spacing window/size)
    from a blurred window x window patch, bias/gain normalised."""
    g = gaussian_filter(to_gray(im), 2)
    step = window // size
    offs = (np.arange(size) - size / 2 + 0.5) * step
    oy, ox = np.meshgrid(offs, offs, indexing='ij')
    desc = []
    for x, y in pts:
        p = map_coordinates(g, [oy + y, ox + x], order=1, mode='nearest')
        desc.append(((p - p.mean()) / (p.std() + 1e-8)).ravel())
    return np.array(desc)


def match_features(d1, d2, ratio=0.6):
    """Lowe ratio test on first/second nearest neighbour. Returns index pairs."""
    D = dist2(d1, d2)
    order = np.argsort(D, axis=1)
    i = np.arange(len(d1))
    nn1, nn2 = D[i, order[:, 0]], D[i, order[:, 1]]
    ok = np.sqrt(np.maximum(nn1, 0)) < ratio * np.sqrt(np.maximum(nn2, 0))
    return np.c_[i[ok], order[ok, 0]]


def ransac_homography(p1, p2, n_iter=5000, eps=3.0, seed=0):
    """4-point RANSAC. Returns (H refit on all inliers, inlier mask)."""
    rng = np.random.default_rng(seed)
    best = np.zeros(len(p1), bool)
    for _ in range(n_iter):
        idx = rng.choice(len(p1), 4, replace=False)
        H = computeH(p1[idx], p2[idx])
        if not np.all(np.isfinite(H)):
            continue
        err = np.linalg.norm(apply_H(H, p1) - p2, axis=1)
        inl = err < eps
        if inl.sum() > best.sum():
            best = inl
    return computeH(p1[best], p2[best]), best
