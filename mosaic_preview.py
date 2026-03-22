"""
mosaic_preview.py — Render isolated PNGs as a grid-cell mosaic video.

Reads RGBA PNGs from the isolated/ directory and produces a video where
visible regions are rendered as bordered square cells on a white background.

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


def render_mosaic(image_rgba: np.ndarray, cell_size: int, seed: int, palette: list, border_color: tuple = (0, 0, 0)) -> np.ndarray:
    """Render occupied cells as white squares with a colored border."""
    h, w = image_rgba.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    alpha = image_rgba[:, :, 3]

    for row in range(0, h, cell_size):
        for col in range(0, w, cell_size):
            cell_alpha = alpha[row : row + cell_size, col : col + cell_size]
            if cell_alpha.mean() < 30:
                continue

            cx = col + cell_size // 2
            cy = row + cell_size // 2
            radius = int(cell_size * 0.3)
            cv2.circle(canvas, (cx, cy), radius, border_color, -1, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Geometric mosaic preview from isolated PNGs")
    parser.add_argument("--input", default="isolated", help="Directory of RGBA PNGs")
    parser.add_argument("--output", default="preview_mosaic.mp4", help="Output video path")
    parser.add_argument("--cell-size", type=int, default=10, help="Grid cell size in pixels")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--color", default="#000000", metavar="HEX",
                        help="Hex color for cell borders (default #000000)")
    parser.add_argument("--frame-stack", default=None, metavar="PATH",
                        help="Output path for a composite PNG of all mosaic frames stacked")
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

    border_rgb = _hex_to_rgb(args.color)
    border_bgr = (border_rgb[2], border_rgb[1], border_rgb[0])
    stack = np.full((h, w, 3), 255, dtype=np.uint8) if args.frame_stack else None

    print(f"Processing {len(png_paths)} frames (cell_size={args.cell_size})...")
    for i, path in enumerate(png_paths):
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue
        # Ensure RGBA
        if frame.shape[2] == 3:
            alpha_ch = np.full((h, w, 1), 255, dtype=np.uint8)
            frame = np.concatenate([frame, alpha_ch], axis=2)

        mosaic = render_mosaic(frame, args.cell_size, seed=i, palette=[], border_color=border_bgr)
        writer.write(mosaic)

        if stack is not None:
            mask = np.any(mosaic != 255, axis=2)
            stack[mask] = mosaic[mask]

        if (i + 1) % 50 == 0 or i == len(png_paths) - 1:
            print(f"  {i + 1}/{len(png_paths)}")

    writer.release()
    print(f"Saved {args.output}")

    if stack is not None:
        cv2.imwrite(args.frame_stack, stack)
        print(f"Saved frame stack {args.frame_stack}")


if __name__ == "__main__":
    main()
