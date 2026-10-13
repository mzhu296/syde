"""Create a concise coarse-to-fine alignment GIF for one glass plate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import load_image, shift_image, split_channels
from visualize_alignment import align_with_history


SCRIPT_DIR = Path(__file__).resolve().parent


def to_pil(image):
    pixels = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def fit_image(image, size):
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    return fitted


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def crop_fixed(image, border):
    if border <= 0:
        return image
    return image[border:-border, border:-border]


def make_frame(before, current, heading, detail, step, total):
    width, height = 1100, 650
    canvas = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30, bold=True)
    label_font = load_font(21, bold=True)
    detail_font = load_font(18)
    small_font = load_font(15)

    draw.text((40, 24), "01007a: coarse-to-fine pyramid alignment", fill="#102a43", font=title_font)
    draw.text((40, 70), heading, fill="#075985", font=label_font)
    draw.text((40, 102), detail, fill="#52606d", font=detail_font)

    left = fit_image(before, (495, 440))
    right = fit_image(current, (495, 440))
    left_xy = (35 + (495 - left.width) // 2, 145 + (440 - left.height) // 2)
    right_xy = (570 + (495 - right.width) // 2, 145 + (440 - right.height) // 2)
    canvas.paste(left, left_xy)
    canvas.paste(right, right_xy)

    draw.rectangle((34, 144, 531, 586), outline="#cbd5e1", width=2)
    draw.rectangle((569, 144, 1066, 586), outline="#0f766e", width=3)
    draw.text((35, 600), "Unaligned channel stack", fill="#52606d", font=small_font)
    draw.text((570, 600), "Current alignment estimate", fill="#0f766e", font=small_font)

    dot_y = 626
    start_x = 910
    for index in range(total):
        x = start_x + index * 24
        fill = "#0f766e" if index <= step else "#d9e2ec"
        draw.ellipse((x, dot_y - 6, x + 12, dot_y + 6), fill=fill)

    return canvas


def create_gif(image_path, output_path):
    image = load_image(image_path)
    B, G, R = split_channels(image)
    g_dx, g_dy, green_history, _, _ = align_with_history(G, B, "Green")
    r_dx, r_dy, red_history, _, _ = align_with_history(R, B, "Red")

    final_border = 20 + max(abs(g_dx), abs(g_dy), abs(r_dx), abs(r_dy))
    unaligned_array = crop_fixed(np.dstack([R, G, B]), final_border)
    before = to_pil(unaligned_array)

    frames = []
    opening = make_frame(
        before,
        before,
        "Start: channels stacked without alignment",
        "Green and red are visibly displaced from the fixed blue reference.",
        0,
        len(green_history) + 1,
    )
    frames.append(opening)

    for index, (green_step, red_step) in enumerate(zip(green_history, red_history), start=1):
        level = green_step["level"]
        green_offset = (
            green_step["full_resolution_dx"],
            green_step["full_resolution_dy"],
        )
        red_offset = (
            red_step["full_resolution_dx"],
            red_step["full_resolution_dy"],
        )
        aligned = np.dstack([
            shift_image(R, *red_offset),
            shift_image(G, *green_offset),
            B,
        ])
        aligned = crop_fixed(aligned, final_border)
        resolution = f"{green_step['width']} × {green_step['height']}"
        heading = f"Pyramid level {level}: {resolution} search image"
        detail = f"Projected offsets — Green {green_offset}; Red {red_offset}"
        frames.append(make_frame(
            before,
            to_pil(aligned),
            heading,
            detail,
            index,
            len(green_history) + 1,
        ))

    durations = [1700] + [1050] * (len(frames) - 2) + [2600]
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
    return (g_dx, g_dy), (r_dx, r_dy), len(frames)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR / "data" / "01007a.jpg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "visualizations" / "01007a" / "01007a_pyramid_alignment.gif",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image.resolve()
    output_path = args.output.resolve()
    green_offset, red_offset, frame_count = create_gif(image_path, output_path)
    print(f"Input: {image_path}")
    print(f"Frames: {frame_count}")
    print(f"Final green offset: {green_offset}")
    print(f"Final red offset: {red_offset}")
    print(f"Saved GIF to: {output_path}")


if __name__ == "__main__":
    main()
