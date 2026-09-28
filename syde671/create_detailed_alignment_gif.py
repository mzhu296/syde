"""Create a detailed end-to-end alignment walkthrough GIF for 01007a."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import load_image, shift_image, split_channels
from visualize_alignment import align_with_history


SCRIPT_DIR = Path(__file__).resolve().parent
CANVAS_SIZE = (1200, 760)
BACKGROUND = "#f8fafc"
INK = "#102a43"
MUTED = "#52606d"
ACCENT = "#075985"
GREEN = "#0f766e"


def font(size, bold=False):
    choices = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in choices:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_pil(array, mode="RGB"):
    pixels = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, mode=mode)


def fit(image, size):
    result = image.copy().convert("RGB")
    result.thumbnail(size, Image.Resampling.LANCZOS)
    return result


def base_frame(step, total, title, subtitle):
    canvas = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 28), "01007a: automatic glass-plate alignment", fill=INK, font=font(31, True))
    draw.text((42, 78), title, fill=ACCENT, font=font(23, True))
    draw.text((42, 116), subtitle, fill=MUTED, font=font(18))
    draw.text((1080, 34), f"{step}/{total}", fill=MUTED, font=font(17, True))
    progress_left, progress_right, y = 42, 1158, 722
    draw.line((progress_left, y, progress_right, y), fill="#d9e2ec", width=5)
    completed = progress_left + (progress_right - progress_left) * (step - 1) / (total - 1)
    draw.line((progress_left, y, completed, y), fill=GREEN, width=5)
    return canvas


def paste_center(canvas, image, box, border="#cbd5e1"):
    x, y, width, height = box
    image = fit(image, (width, height))
    px = x + (width - image.width) // 2
    py = y + (height - image.height) // 2
    canvas.paste(image, (px, py))
    ImageDraw.Draw(canvas).rectangle((x, y, x + width, y + height), outline=border, width=2)


def labelled_panel(canvas, image, box, label, border="#cbd5e1"):
    x, y, width, height = box
    paste_center(canvas, image, box, border)
    ImageDraw.Draw(canvas).text((x, y + height + 9), label, fill=MUTED, font=font(16, True))


def raw_plate_frame(raw, step, total):
    canvas = base_frame(step, total, "1. Load the glass-plate image",
                        "One grayscale plate stores blue, green, and red exposures vertically.")
    labelled_panel(canvas, to_pil(raw, mode="L"), (350, 158, 500, 500),
                   f"Input plate · {raw.shape[1]} × {raw.shape[0]} pixels")
    return canvas


def split_frame(B, G, R, step, total):
    canvas = base_frame(step, total, "2. Split the plate into B / G / R",
                        "Each channel receives one equal-height third of the source image.")
    boxes = [(42, 175, 350, 430), (425, 175, 350, 430), (808, 175, 350, 430)]
    labels = ["Blue reference", "Green moving channel", "Red moving channel"]
    colors = ["#2563eb", "#0f766e", "#dc2626"]
    for image, box, label, color in zip((B, G, R), boxes, labels, colors):
        labelled_panel(canvas, to_pil(image, mode="L"), box, label, color)
    return canvas


def rgb_frame(rgb, step, total, title, subtitle, label):
    canvas = base_frame(step, total, title, subtitle)
    labelled_panel(canvas, to_pil(rgb), (210, 165, 780, 490), label, GREEN)
    return canvas


def imported_figure_frame(path, step, total, title, subtitle):
    canvas = base_frame(step, total, title, subtitle)
    labelled_panel(canvas, Image.open(path), (90, 155, 1020, 520), path.stem.replace("_", " "))
    return canvas


def alignment_stage_frame(before, current, step, total, title, subtitle, detail):
    canvas = base_frame(step, total, title, subtitle)
    labelled_panel(canvas, before, (42, 175, 535, 430), "Unaligned", "#cbd5e1")
    labelled_panel(canvas, current, (623, 175, 535, 430), detail, GREEN)
    return canvas


def crop(array, border):
    return array[border:-border, border:-border]


def create_detailed_gif(image_path, output_path, assets_dir):
    raw = load_image(image_path)
    B, G, R = split_channels(raw)
    g_dx, g_dy, green_history, _, _ = align_with_history(G, B, "Green")
    r_dx, r_dy, red_history, _, _ = align_with_history(R, B, "Red")
    border = 20 + max(abs(g_dx), abs(g_dy), abs(r_dx), abs(r_dy))

    unaligned = crop(np.dstack([R, G, B]), border)
    l2 = crop(np.dstack([
        shift_image(R, -9, 15), shift_image(G, -4, 15), B
    ]), border)
    ncc_single = l2.copy()
    before = to_pil(unaligned)

    total = 8 + len(green_history)
    frames = [
        raw_plate_frame(raw, 1, total),
        split_frame(B, G, R, 2, total),
        rgb_frame(
            unaligned, 3, total, "3. Stack channels before alignment",
            "Misregistered channel edges appear as strong red, green, and blue ghosts.",
            "Unaligned RGB composite",
        ),
        imported_figure_frame(
            assets_dir / "01007a_l2_heatmap.png", 4, total,
            "4. Single-scale L2 search",
            "At the coarsest resolution, every integer displacement in [-15, 15]² is scored.",
        ),
        rgb_frame(
            l2, 5, total, "5. Apply the best single-scale L2 offsets",
            "At full resolution, the ±15-pixel window is too small for the true displacement.",
            "L2 result · G (-4, 15), R (-9, 15)",
        ),
        imported_figure_frame(
            assets_dir / "01007a_ncc_heatmap.png", 6, total,
            "6. Single-scale normalized cross-correlation",
            "NCC selects the largest normalized similarity score and tolerates brightness changes.",
        ),
        rgb_frame(
            ncc_single, 7, total, "7. Apply the best single-scale NCC offsets",
            "NCC reaches the same bounded offsets, confirming that a wider search strategy is needed.",
            "NCC result · G (-4, 15), R (-9, 15)",
        ),
        imported_figure_frame(
            assets_dir / "01007a_pyramid_levels.png", 8, total,
            "8. Build Gaussian pyramids",
            "Blur and downsample B, G, and R from full resolution to a 238 × 203 coarse level.",
        ),
    ]

    for index, (green_step, red_step) in enumerate(zip(green_history, red_history), start=9):
        level = green_step["level"]
        green_offset = (green_step["full_resolution_dx"], green_step["full_resolution_dy"])
        red_offset = (red_step["full_resolution_dx"], red_step["full_resolution_dy"])
        aligned = crop(np.dstack([
            shift_image(R, *red_offset),
            shift_image(G, *green_offset),
            B,
        ]), border)
        search_text = "global ±15 search" if level == green_history[0]["level"] else "local ±2 refinement"
        frames.append(alignment_stage_frame(
            before,
            to_pil(aligned),
            index,
            total,
            f"9. Pyramid NCC · level {level}",
            f"{green_step['width']} × {green_step['height']} search image · {search_text}",
            f"Projected G {green_offset} · R {red_offset}",
        ))

    durations = [1700, 1700, 1500, 2100, 1600, 2100, 1600, 2100]
    durations += [1250] * (len(green_history) - 1) + [3000]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return len(frames), (g_dx, g_dy), (r_dx, r_dy)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path,
                        default=SCRIPT_DIR / "data" / "01007a.jpg")
    parser.add_argument("--output", type=Path,
                        default=SCRIPT_DIR / "visualizations" / "01007a" /
                        "01007a_detailed_alignment.gif")
    return parser.parse_args()


def main():
    args = parse_args()
    image = args.image.resolve()
    output = args.output.resolve()
    assets = SCRIPT_DIR / "visualizations" / "01007a"
    frame_count, green, red = create_detailed_gif(image, output, assets)
    print(f"Frames: {frame_count}")
    print(f"Final green offset: {green}")
    print(f"Final red offset: {red}")
    print(f"Saved GIF to: {output}")


if __name__ == "__main__":
    main()
