import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

from pathlib import Path
import csv
import time

# ============================================================
# 1. Load image
# ============================================================

def load_image(path):
    image = iio.imread(path)

    # Convert integer image to float [0, 1]
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        image = image.astype(np.float32) / max_value
    else:
        image = image.astype(np.float32)

    return image


# ============================================================
# 2. Split glass plate into B, G, R
# ============================================================

def split_channels(image):
    height = image.shape[0]

    channel_height = height // 3

    B = image[
        0:channel_height,
        :
    ]

    G = image[
        channel_height:2 * channel_height,
        :
    ]

    R = image[
        2 * channel_height:3 * channel_height,
        :
    ]

    return B, G, R


# ============================================================
# 3. Shift image
# ============================================================

def shift_image(image, dx, dy):
    """
    dx > 0 : right
    dx < 0 : left

    dy > 0 : down
    dy < 0 : up
    """

    return np.roll(
        image,
        shift=(dy, dx),
        axis=(0, 1)
    )


# ============================================================
# 4. Crop border
# ============================================================

def crop_border(image, border=20):
    return image[
        border:-border,
        border:-border
    ]


# ============================================================
# 5. L2 distance
# ============================================================

def l2_distance(image1, image2):
    difference = image1 - image2

    score = np.sqrt(
        np.sum(difference ** 2)
    )

    return score


# ============================================================
# 6. NCC
# ============================================================

def ncc(image1, image2):
    """
    Normalized Cross-Correlation.

    Assignment definition:
    dot product between normalized image vectors.
    """

    a = image1.flatten()
    b = image2.flatten()

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return -1

    a = a / a_norm
    b = b / b_norm

    return np.dot(a, b)


# ============================================================
# 7. Single-scale L2 alignment
# ============================================================

def align_l2(
    moving,
    reference,
    search_range=15,
    border=20
):

    best_score = float("inf")

    best_dx = 0
    best_dy = 0

    reference_crop = crop_border(
        reference,
        border
    )

    for dy in range(
        -search_range,
        search_range + 1
    ):

        for dx in range(
            -search_range,
            search_range + 1
        ):

            shifted = shift_image(
                moving,
                dx,
                dy
            )

            shifted_crop = crop_border(
                shifted,
                border
            )

            score = l2_distance(
                shifted_crop,
                reference_crop
            )

            # Smaller L2 = better
            if score < best_score:

                best_score = score

                best_dx = dx
                best_dy = dy

    return best_dx, best_dy, best_score


# ============================================================
# 8. Single-scale NCC alignment
# ============================================================

def align_ncc(
    moving,
    reference,
    search_range=15,
    border=20
):

    best_score = -float("inf")

    best_dx = 0
    best_dy = 0

    reference_crop = crop_border(
        reference,
        border
    )

    for dy in range(
        -search_range,
        search_range + 1
    ):

        for dx in range(
            -search_range,
            search_range + 1
        ):

            shifted = shift_image(
                moving,
                dx,
                dy
            )

            shifted_crop = crop_border(
                shifted,
                border
            )

            score = ncc(
                shifted_crop,
                reference_crop
            )

            # Larger NCC = better
            if score > best_score:

                best_score = score

                best_dx = dx
                best_dy = dy

    return best_dx, best_dy, best_score


# ============================================================
# 9. Display RGB
# ============================================================

def show_rgb(
    image,
    title,
    border=20
):

    # Crop display border because np.roll wraps pixels
    cropped = image[
        border:-border,
        border:-border,
        :
    ]

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        np.clip(
            cropped,
            0,
            1
        )
    )

    plt.title(title)

    plt.axis("off")

    plt.show()


# ============================================================
# 10. Gaussian/binomial blur
# ============================================================

def blur_image(image):
    """
    Apply:

        [1, 4, 6, 4, 1] / 16

    horizontally and vertically.
    """

    kernel = np.array(
        [1, 4, 6, 4, 1],
        dtype=np.float32
    ) / 16.0

    height, width = image.shape

    # --------------------------------------------------------
    # Horizontal blur
    # --------------------------------------------------------

    padded = np.pad(
        image,
        ((0, 0), (2, 2)),
        mode="reflect"
    )

    horizontal = np.zeros_like(
        image
    )

    for i in range(5):

        horizontal += (
            kernel[i]
            * padded[
                :,
                i:i + width
            ]
        )

    # --------------------------------------------------------
    # Vertical blur
    # --------------------------------------------------------

    padded = np.pad(
        horizontal,
        ((2, 2), (0, 0)),
        mode="reflect"
    )

    blurred = np.zeros_like(
        image
    )

    for i in range(5):

        blurred += (
            kernel[i]
            * padded[
                i:i + height,
                :
            ]
        )

    return blurred


# ============================================================
# 11. Downsample by factor of 2
# ============================================================

def downsample(image):

    # Blur BEFORE subsampling
    blurred = blur_image(image)

    # Keep every second row and column
    smaller = blurred[
        ::2,
        ::2
    ]

    return smaller


# ============================================================
# 12. Build Gaussian pyramid
# ============================================================

def build_pyramid(
    image,
    min_size=150
):

    # Level 0 = full-resolution image
    pyramid = [image]

    while (
        min(pyramid[-1].shape)
        >= 2 * min_size
    ):

        smaller = downsample(
            pyramid[-1]
        )

        pyramid.append(
            smaller
        )

    return pyramid


# ============================================================
# 13. Search around an existing displacement
# ============================================================

def search_alignment(
    moving,
    reference,
    center_dx=0,
    center_dy=0,
    search_range=15,
    metric="ncc",
    border=10
):

    min_dx = center_dx - search_range
    max_dx = center_dx + search_range

    min_dy = center_dy - search_range
    max_dy = center_dy + search_range

    # Largest possible displacement being tested
    largest_shift = max(
        abs(min_dx),
        abs(max_dx),
        abs(min_dy),
        abs(max_dy)
    )

    # Avoid np.roll wrapped pixels during scoring
    margin = (
        border
        + largest_shift
    )

    if (
        2 * margin
        >= min(reference.shape)
    ):
        raise ValueError(
            "Image too small for this search range."
        )

    reference_crop = reference[
        margin:-margin,
        margin:-margin
    ]

    # --------------------------------------------------------
    # Initialize score
    # --------------------------------------------------------

    if metric == "l2":

        best_score = float("inf")

    elif metric == "ncc":

        best_score = -float("inf")

    else:

        raise ValueError(
            "metric must be 'l2' or 'ncc'"
        )

    best_dx = center_dx
    best_dy = center_dy

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    for dy in range(
        min_dy,
        max_dy + 1
    ):

        for dx in range(
            min_dx,
            max_dx + 1
        ):

            shifted = shift_image(
                moving,
                dx,
                dy
            )

            shifted_crop = shifted[
                margin:-margin,
                margin:-margin
            ]

            # ----------------------------------------------
            # L2
            # ----------------------------------------------

            if metric == "l2":

                score = l2_distance(
                    shifted_crop,
                    reference_crop
                )

                if score < best_score:

                    best_score = score

                    best_dx = dx
                    best_dy = dy

            # ----------------------------------------------
            # NCC
            # ----------------------------------------------

            else:

                score = ncc(
                    shifted_crop,
                    reference_crop
                )

                if score > best_score:

                    best_score = score

                    best_dx = dx
                    best_dy = dy

    return (
        best_dx,
        best_dy,
        best_score
    )


# ============================================================
# 14. Coarse-to-fine pyramid alignment
# ============================================================

def align_pyramid(
    moving,
    reference,
    metric="ncc",
    coarse_search=15,
    refine_search=2,
    min_size=150
):

    moving_pyramid = build_pyramid(
        moving,
        min_size=min_size
    )

    reference_pyramid = build_pyramid(
        reference,
        min_size=min_size
    )

    print(
        "\nPyramid levels:",
        len(reference_pyramid)
    )

    dx = 0
    dy = 0

    score = None

    # Start from smallest image
    for level in reversed(
        range(
            len(reference_pyramid)
        )
    ):

        moving_level = (
            moving_pyramid[level]
        )

        reference_level = (
            reference_pyramid[level]
        )

        print(
            "\nLevel:",
            level
        )

        print(
            "Image size:",
            reference_level.shape
        )

        # ----------------------------------------------------
        # Coarsest level
        # ----------------------------------------------------

        if (
            level
            == len(reference_pyramid) - 1
        ):

            dx, dy, score = search_alignment(
                moving_level,
                reference_level,
                center_dx=0,
                center_dy=0,
                search_range=coarse_search,
                metric=metric
            )

        # ----------------------------------------------------
        # Finer level
        # ----------------------------------------------------

        else:

            # Image doubled in size
            # therefore displacement doubles
            dx *= 2
            dy *= 2

            print(
                "Predicted displacement:",
                (dx, dy)
            )

            # Refine only near predicted position
            dx, dy, score = search_alignment(
                moving_level,
                reference_level,
                center_dx=dx,
                center_dy=dy,
                search_range=refine_search,
                metric=metric
            )

        print(
            "Best displacement:",
            (dx, dy)
        )

    return dx, dy, score

# ============================================================
# Save RGB image
# ============================================================

def save_rgb(image, path, border=20):
    """
    Crop borders, convert [0,1] float image to uint8,
    and save as JPG.
    """

    height, width, _ = image.shape

    # Prevent accidentally cropping away the whole image
    safe_border = min(
        border,
        max(0, height // 4),
        max(0, width // 4)
    )

    if safe_border > 0:
        image = image[
            safe_border:-safe_border,
            safe_border:-safe_border,
            :
        ]

    # Keep RGB values valid
    image = np.clip(
        image,
        0,
        1
    )

    # Convert float [0,1] -> uint8 [0,255]
    image_uint8 = (
        image * 255
    ).round().astype(np.uint8)

    iio.imwrite(
        path,
        image_uint8
    )


# ============================================================
# Determine crop needed after shifting
# ============================================================

def shift_crop_border(
    g_dx,
    g_dy,
    r_dx,
    r_dy,
    base_border=20
):
    """
    np.roll wraps pixels around the image.

    Crop enough pixels to remove the wrapped regions.
    """

    largest_shift = max(
        abs(g_dx),
        abs(g_dy),
        abs(r_dx),
        abs(r_dy)
    )

    return (
        base_border
        + largest_shift
    )


# ============================================================
# Process one glass plate
# ============================================================

def process_one_image(
    image_path,
    output_dir
):

    start_time = time.perf_counter()

    filename = image_path.name
    stem = image_path.stem

    print("\n")
    print("=" * 60)
    print("Processing:", filename)
    print("=" * 60)

    # --------------------------------------------------------
    # Load + split
    # --------------------------------------------------------

    image = load_image(
        str(image_path)
    )

    B, G, R = split_channels(
        image
    )

    print(
        "Channel size:",
        B.shape
    )

    # ========================================================
    # 1. Unaligned
    # ========================================================

    rgb_unaligned = np.dstack([
        R,
        G,
        B
    ])

    save_rgb(
        rgb_unaligned,
        output_dir
        / f"{stem}_unaligned.jpg",
        border=20
    )

    # ========================================================
    # 2. Single-scale L2
    # ========================================================

    g_dx_l2, g_dy_l2, g_score_l2 = align_l2(
        G,
        B
    )

    r_dx_l2, r_dy_l2, r_score_l2 = align_l2(
        R,
        B
    )

    G_l2 = shift_image(
        G,
        g_dx_l2,
        g_dy_l2
    )

    R_l2 = shift_image(
        R,
        r_dx_l2,
        r_dy_l2
    )

    rgb_l2 = np.dstack([
        R_l2,
        G_l2,
        B
    ])

    l2_crop = shift_crop_border(
        g_dx_l2,
        g_dy_l2,
        r_dx_l2,
        r_dy_l2
    )

    save_rgb(
        rgb_l2,
        output_dir
        / f"{stem}_l2.jpg",
        border=l2_crop
    )

    print(
        "L2 G:",
        (g_dx_l2, g_dy_l2),
        "R:",
        (r_dx_l2, r_dy_l2)
    )

    # ========================================================
    # 3. Single-scale NCC
    # ========================================================

    g_dx_ncc, g_dy_ncc, g_score_ncc = align_ncc(
        G,
        B
    )

    r_dx_ncc, r_dy_ncc, r_score_ncc = align_ncc(
        R,
        B
    )

    G_ncc = shift_image(
        G,
        g_dx_ncc,
        g_dy_ncc
    )

    R_ncc = shift_image(
        R,
        r_dx_ncc,
        r_dy_ncc
    )

    rgb_ncc = np.dstack([
        R_ncc,
        G_ncc,
        B
    ])

    ncc_crop = shift_crop_border(
        g_dx_ncc,
        g_dy_ncc,
        r_dx_ncc,
        r_dy_ncc
    )

    save_rgb(
        rgb_ncc,
        output_dir
        / f"{stem}_ncc.jpg",
        border=ncc_crop
    )

    print(
        "NCC G:",
        (g_dx_ncc, g_dy_ncc),
        "R:",
        (r_dx_ncc, r_dy_ncc)
    )

    # ========================================================
    # 4. Pyramid NCC
    # ========================================================

    print(
        "\nPyramid: Green -> Blue"
    )

    g_dx_pyr, g_dy_pyr, g_score_pyr = (
        align_pyramid(
            G,
            B,
            metric="ncc"
        )
    )

    print(
        "\nPyramid: Red -> Blue"
    )

    r_dx_pyr, r_dy_pyr, r_score_pyr = (
        align_pyramid(
            R,
            B,
            metric="ncc"
        )
    )

    G_pyr = shift_image(
        G,
        g_dx_pyr,
        g_dy_pyr
    )

    R_pyr = shift_image(
        R,
        r_dx_pyr,
        r_dy_pyr
    )

    rgb_pyr = np.dstack([
        R_pyr,
        G_pyr,
        B
    ])

    pyramid_crop = shift_crop_border(
        g_dx_pyr,
        g_dy_pyr,
        r_dx_pyr,
        r_dy_pyr
    )

    save_rgb(
        rgb_pyr,
        output_dir
        / f"{stem}_pyramid_ncc.jpg",
        border=pyramid_crop
    )

    print(
        "Pyramid G:",
        (g_dx_pyr, g_dy_pyr),
        "R:",
        (r_dx_pyr, r_dy_pyr)
    )

    # ========================================================
    # Runtime
    # ========================================================

    runtime = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Runtime: {runtime:.2f} seconds"
    )

    # ========================================================
    # Information for CSV
    # ========================================================

    result = {

        "filename":
            filename,

        "channel_height":
            B.shape[0],

        "channel_width":
            B.shape[1],

        # L2
        "l2_green_dx":
            g_dx_l2,

        "l2_green_dy":
            g_dy_l2,

        "l2_red_dx":
            r_dx_l2,

        "l2_red_dy":
            r_dy_l2,

        "l2_green_score":
            float(g_score_l2),

        "l2_red_score":
            float(r_score_l2),

        # NCC
        "ncc_green_dx":
            g_dx_ncc,

        "ncc_green_dy":
            g_dy_ncc,

        "ncc_red_dx":
            r_dx_ncc,

        "ncc_red_dy":
            r_dy_ncc,

        "ncc_green_score":
            float(g_score_ncc),

        "ncc_red_score":
            float(r_score_ncc),

        # Pyramid NCC
        "pyramid_green_dx":
            g_dx_pyr,

        "pyramid_green_dy":
            g_dy_pyr,

        "pyramid_red_dx":
            r_dx_pyr,

        "pyramid_red_dy":
            r_dy_pyr,

        "pyramid_green_score":
            float(g_score_pyr),

        "pyramid_red_score":
            float(r_score_pyr),

        "runtime_seconds":
            runtime,

        "status":
            "OK",

        "error":
            ""
    }

    return result
# ============================================================
# MAIN — process every JPG
# ============================================================

def main():

    data_dir = Path(
        "data"
    )

    output_dir = Path(
        "outputs"
    )

    # Create outputs/ if it does not exist
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find all JPG files
    # --------------------------------------------------------

    image_paths = sorted(
        data_dir.glob("*.jpg")
    )

    print(
        "Found",
        len(image_paths),
        "images"
    )

    if len(image_paths) == 0:

        print(
            "No JPG images found in data/"
        )

        return

    results = []

    # --------------------------------------------------------
    # CSV columns
    # --------------------------------------------------------

    csv_fields = [

        "filename",

        "channel_height",
        "channel_width",

        "l2_green_dx",
        "l2_green_dy",

        "l2_red_dx",
        "l2_red_dy",

        "l2_green_score",
        "l2_red_score",

        "ncc_green_dx",
        "ncc_green_dy",

        "ncc_red_dx",
        "ncc_red_dy",

        "ncc_green_score",
        "ncc_red_score",

        "pyramid_green_dx",
        "pyramid_green_dy",

        "pyramid_red_dx",
        "pyramid_red_dy",

        "pyramid_green_score",
        "pyramid_red_score",

        "runtime_seconds",

        "status",
        "error"
    ]

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(
            f"\n[{index}/{len(image_paths)}]"
        )

        try:

            result = process_one_image(
                image_path,
                output_dir
            )

        except Exception as error:

            print(
                "FAILED:",
                image_path.name
            )

            print(
                error
            )

            # Blank row except error information
            result = {
                field: ""
                for field in csv_fields
            }

            result["filename"] = (
                image_path.name
            )

            result["status"] = (
                "FAILED"
            )

            result["error"] = (
                str(error)
            )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    csv_path = (
        output_dir
        / "results.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINISHED")
    print("=" * 60)

    print(
        "Processed:",
        len(image_paths),
        "images"
    )

    print(
        "Images saved to:",
        output_dir
    )

    print(
        "Offsets saved to:",
        csv_path
    )

    print("\nPyramid offsets:")

    for result in results:

        if result["status"] == "OK":

            print(
                result["filename"],
                "| G:",
                (
                    result[
                        "pyramid_green_dx"
                    ],
                    result[
                        "pyramid_green_dy"
                    ]
                ),
                "| R:",
                (
                    result[
                        "pyramid_red_dx"
                    ],
                    result[
                        "pyramid_red_dy"
                    ]
                )
            )

        else:

            print(
                result["filename"],
                "| FAILED"
            )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
