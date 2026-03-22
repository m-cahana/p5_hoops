"""
mosaic_preview.py — Render isolated PNGs as a wandering-path mosaic video.

Reads RGBA PNGs from the isolated/ directory and produces a video where
the visible silhouette is filled with a single wandering path that makes
90-degree turns at random intervals (Hamming cover style).

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


# Cardinal directions: right, down, left, up
_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def _walk_path(canvas, mask, x, y, steps, rng, palette, cell_size):
    """Walk a single wandering path for `steps` pixel-steps, drawing onto canvas."""
    h, w = mask.shape
    line_thickness = max(1, cell_size // 5)
    min_seg = cell_size // 2
    max_seg = cell_size * 3

    chosen = palette[rng.randint(len(palette))]
    color_bgr = (chosen[2], chosen[1], chosen[0])

    dir_idx = rng.randint(4)
    dx, dy = _DIRS[dir_idx]
    seg_len = rng.randint(min_seg, max_seg + 1)
    seg_walked = 0
    walked = 0
    prev_x, prev_y = x, y
    stuck_count = 0

    while walked < steps:
        nx, ny = x + dx, y + dy
        in_bounds = 0 <= nx < w and 0 <= ny < h
        in_mask = in_bounds and mask[ny, nx]

        if in_mask and seg_walked < seg_len:
            x, y = nx, ny
            seg_walked += 1
            walked += 1
            stuck_count = 0
        else:
            # Draw accumulated segment
            if (prev_x, prev_y) != (x, y):
                cv2.line(canvas, (prev_x, prev_y), (x, y), color_bgr, line_thickness)

            # Try turning 90 degrees
            turn = rng.choice([-1, 1])
            new_dir = (dir_idx + turn) % 4
            alt_dir = (dir_idx - turn) % 4
            ndx, ndy = _DIRS[new_dir]
            adx, ady = _DIRS[alt_dir]
            can_turn = (0 <= x + ndx < w and 0 <= y + ndy < h and mask[y + ndy, x + ndx])
            can_alt = (0 <= x + adx < w and 0 <= y + ady < h and mask[y + ady, x + adx])

            if can_turn:
                dir_idx = new_dir
            elif can_alt:
                dir_idx = alt_dir
            elif in_mask:
                # Can't turn, keep going forward
                dx, dy = _DIRS[dir_idx]
                seg_len = seg_walked + rng.randint(min_seg, max_seg + 1)
                seg_walked = 0
                prev_x, prev_y = x, y
                continue
            else:
                # Stuck — bail out of this path
                stuck_count += 1
                if stuck_count > 5:
                    break
                # Try reversing
                dir_idx = (dir_idx + 2) % 4

            dx, dy = _DIRS[dir_idx]
            seg_len = rng.randint(min_seg, max_seg + 1)
            seg_walked = 0
            prev_x, prev_y = x, y

    # Draw final segment
    if (prev_x, prev_y) != (x, y):
        cv2.line(canvas, (prev_x, prev_y), (x, y), color_bgr, line_thickness)


def render_mosaic(image_rgba: np.ndarray, cell_size: int, seed: int, palette: list, density: float = 0.15) -> np.ndarray:
    """Render each connected form as its own wandering path with 90-degree turns.

    Each isolated shape (connected component) in the alpha mask gets a single
    path in its own palette color, walking through that form only.
    """
    h, w = image_rgba.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    alpha = image_rgba[:, :, 3]
    mask = (alpha >= 30).astype(np.uint8)
    if not mask.any():
        return canvas

    rng = np.random.RandomState(seed)

    # Find connected components (each isolated form)
    n_labels, labels = cv2.connectedComponents(mask)

    for label in range(1, n_labels):
        component_mask = labels == label
        ys, xs = np.where(component_mask)
        if len(ys) == 0:
            continue

        n_steps = int(len(ys) * density)
        if n_steps < 2:
            continue

        start_idx = rng.randint(len(ys))
        x, y = int(xs[start_idx]), int(ys[start_idx])
        _walk_path(canvas, component_mask, x, y, n_steps, rng, palette, cell_size)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Geometric mosaic preview from isolated PNGs")
    parser.add_argument("--input", default="isolated", help="Directory of RGBA PNGs")
    parser.add_argument("--output", default="preview_mosaic.mp4", help="Output video path")
    parser.add_argument("--cell-size", type=int, default=10, help="Grid cell size in pixels")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--color", default=None, metavar="HEX",
                        help="Single hex color for all cells (e.g. #000000). Omit to use PALETTE_HEX.")
    parser.add_argument("--density", type=float, default=0.15,
                        help="Path density as fraction of occupied pixels to walk (default 0.15)")
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

    palette = _build_palette(args.color)
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

        mosaic = render_mosaic(frame, args.cell_size, seed=i, palette=palette, density=args.density)
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
