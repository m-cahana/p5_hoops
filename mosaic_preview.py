"""
mosaic_preview.py — Render isolated PNGs as a geometric mosaic video.

Reads RGBA PNGs from the isolated/ directory and produces a video where
visible regions are rendered as small squares and circles of varying size,
creating a p5-style artistic effect using circles and lines.

Usage:
    python mosaic_preview.py [--input isolated] [--output preview_mosaic.mp4] [--cell-size 8]
"""

import argparse
import glob
import os

import cv2
import numpy as np

# Fixed palette (hex). Used when --color is not passed.
PALETTE_HEX: list[str] = [
    "#FF5000",  # vivid orange
    "#E62800",  # orange-red
    "#C81400",  # deep red
    "#FF8C14",  # warm amber
    "#FFC83C",  # golden yellow
    "#A00A00",  # dark crimson
    "#500500",  # near-black red (shadow anchor)
]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _build_palette(single_color: str | None) -> list[tuple[int, int, int]]:
    if single_color:
        return [_hex_to_rgb(single_color)]
    return [_hex_to_rgb(h) for h in PALETTE_HEX]


DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left


def render_walk(image_rgba: np.ndarray, seed: int, palette: list,
                step: int = 4, num_walkers: int = 200, max_steps: int = 600) -> np.ndarray:
    """Fill visible forms with orthogonal random-walk lines."""
    h, w = image_rgba.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    alpha = image_rgba[:, :, 3]
    mask = alpha > 30  # boolean mask of visible pixels

    # Collect all visible pixel coordinates to seed walkers
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return canvas

    rng = np.random.RandomState(seed)

    for _ in range(num_walkers):
        # Pick a random starting point inside the mask
        idx = rng.randint(len(ys))
        x, y = int(xs[idx]), int(ys[idx])
        dir_idx = rng.randint(4)

        chosen = palette[rng.randint(len(palette))]
        color_bgr = (chosen[2], chosen[1], chosen[0])

        for _ in range(max_steps):
            dx, dy = DIRECTIONS[dir_idx]
            nx, ny = x + dx * step, y + dy * step

            # Stay inside mask — check endpoint and midpoint
            if (0 <= nx < w and 0 <= ny < h and mask[ny, nx]
                    and mask[y + dy * (step // 2), x + dx * (step // 2)]):
                cv2.line(canvas, (x, y), (nx, ny), color_bgr, 1, cv2.LINE_AA)
                x, y = nx, ny
            else:
                # Hit edge — must turn
                pass

            # Randomly turn 90 degrees (or keep going)
            r = rng.random()
            if r < 0.35:
                dir_idx = (dir_idx + 1) % 4
            elif r < 0.7:
                dir_idx = (dir_idx - 1) % 4

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Geometric mosaic preview from isolated PNGs")
    parser.add_argument("--input", default="isolated", help="Directory of RGBA PNGs")
    parser.add_argument("--output", default="preview_mosaic.mp4", help="Output video path")
    parser.add_argument("--cell-size", type=int, default=8, help="Grid cell size in pixels")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--color", default=None, metavar="HEX",
                        help="Single hex color for all cells (e.g. #000000). Omit to use PALETTE_HEX.")
    args = parser.parse_args()

    png_paths = sorted(glob.glob(os.path.join(args.input, "*.png")))
    if not png_paths:
        print(f"No PNGs found in {args.input}/")
        return

    # Read first frame to get dimensions
    sample = cv2.imread(png_paths[0], cv2.IMREAD_UNCHANGED)
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    palette = _build_palette(args.color)

    print(f"Processing {len(png_paths)} frames (cell_size={args.cell_size})...")
    for i, path in enumerate(png_paths):
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue
        # Ensure RGBA
        if frame.shape[2] == 3:
            alpha_ch = np.full((h, w, 1), 255, dtype=np.uint8)
            frame = np.concatenate([frame, alpha_ch], axis=2)

        mosaic = render_walk(frame, seed=i, palette=palette)
        writer.write(mosaic)

        if (i + 1) % 50 == 0 or i == len(png_paths) - 1:
            print(f"  {i + 1}/{len(png_paths)}")

    writer.release()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
