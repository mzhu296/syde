# A.2 notes: how the correspondences were obtained

Image set (also used for Part B): IMG_4826, IMG_4827, IMG_4828. IMG_4828 is the reference;
IMG_4827 -> IMG_4828 and IMG_4826 -> IMG_4828 are the two homographies needed.

## Correspondences: estimated by eye, refined by local correlation (not hand-clicked)

1. **Coarse estimate by eye.** Gridded, zoomed views of the overlap region (grid every
   25-50 px). Distinctive static landmarks (shore rocks, two people on the beach, a beach
   rock, cliff groove top/base, cliff base corner, hill clearing, rock stack) were read off
   the grid in both images. Accuracy about 5-20 px.
2. **Refinement.** For each point a 45x45 grey patch around the image-1 point was
   matched by normalised cross-correlation (`skimage.feature.match_template`) inside a
   +-18 px window around the estimate in image 2. The best match replaced the estimate.
3. **Sanity check against the automatic pipeline.** A reference homography was computed
   with Harris + ANMS (1500) + MOPS descriptors + ratio test (0.7) + RANSAC. Points that
   disagreed with it were dropped (4827 "hill clearing", 7 px; 4826 "rock stack top",
   correlation 0.33) and one pairing was corrected (shore rock -> pointed rock).

**Limitation.** Step 3 means the final points are consistent with, and partly selected
by, the automatic method, so the manual-vs-automatic comparison is not fully independent.
Search windows in step 2 were centred on the eyeballed estimates, not on the reference H.
For a strictly manual result, re-click with `pick_points.py` and save to `points/`.

## Final points

| Pair | n | fit residual (mean / max) |
|---|---|---|
| 4827 -> 4828 | 8 | 1.35 / 2.60 px |
| 4826 -> 4828 | 13 | 1.40 / 3.13 px |

Near-field cliff vs far beach adds a few pixels of parallax.

## Conditioning problem and fix

The 4827 -> 4828 H is sound: it agrees with the automatic H to within 1-5 px at the image
centre and left corners and 47-57 px at the far right corners.

The first 4826 -> 4828 fit (7 points) was ill-conditioned: ~2 px residual, but the image
corners landed thousands of pixels from where the automatic H (45 inliers) sends them,
because all 7 points sat in a ~190 x 260 px patch. Six points were added across the
overlap (hill-clearing left/right edges, boulder pair, boulder, dark rock, tidal pool;
points 8-13). For these the 4826 source point was chosen by eye, but the search window in
4828 was centred on the automatic H's prediction (eyeballing rock-to-rock pairs was not
reliable), then refined by correlation (+-25 px, NCC 0.44-0.81). Rejected: house group,
pale boulder, water boulder (NCC < 0.3 or ambiguous). Result: n = 13, residual 1.40 / 3.13 px,
corners within 3-51 px of the automatic H.

**Consequence.** The 4826 -> 4828 "manual" H is therefore close to the automatic one by
construction; the manual-vs-automatic comparison for that link is weak. Points 1, 8, 9 are
all on the same hill clearing, so the effective spread is smaller than 13 suggests.

Files: `points/p4827_4828.json`, `points/p4826_4828.json`; outputs `A2_*.png`, `A2_*.txt`.
