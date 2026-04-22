"""
invert_preview.py — Render non-isolated frames as a mosaic of squares and circles,
coloring each cell by snapping its average color to the nearest palette entry.

Usage:
    python invert_preview.py --input videos/dunk1/frames --output videos/dunk1/preview_invert.mp4
    python invert_preview.py --input frames --palette '#000000' '#ffffff'
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

from mosaic_preview import _hex_to_rgb, _load_preset, apply_film_grain


BW_PALETTE: list[str] = ["#000000", "#ffffff"]


def _nearest_palette_idx(mean_bgr: np.ndarray, palette_bgr: np.ndarray) -> int:
    diffs = palette_bgr.astype(np.float32) - mean_bgr
    return int(np.argmin((diffs * diffs).sum(axis=1)))


def render_invert(
    image_bgr: np.ndarray,
    cell_size: int,
    seed: int,
    palette_bgr: np.ndarray | None,
    supersample: int = 1,
    squares_pct: int = 0,
    bg_bgr: tuple = (255, 255, 255),
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    s = supersample
    canvas = np.full((h * s, w * s, 3), bg_bgr, dtype=np.uint8)
    rng = np.random.RandomState(seed)

    for row in range(0, h, cell_size):
        for col in range(0, w, cell_size):
            cell = image_bgr[row : row + cell_size, col : col + cell_size]
            if cell.size == 0:
                continue
            mean_bgr = cell.reshape(-1, 3).mean(axis=0)
            if palette_bgr is None:
                color_bgr = tuple(int(v) for v in mean_bgr)
            else:
                idx = _nearest_palette_idx(mean_bgr, palette_bgr)
                color_bgr = tuple(int(v) for v in palette_bgr[idx])

            cx = (col + cell_size // 2) * s
            cy = (row + cell_size // 2) * s
            radius = int(cell_size * 0.3 * s)

            if squares_pct > 0 and rng.randint(100) < squares_pct:
                half = radius
                cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), color_bgr, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (cx, cy), radius, color_bgr, -1, cv2.LINE_AA)

    if s > 1:
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Inverted mosaic: color each cell by source frame color")
    parser.add_argument("--preset", default=None, help="Load palette and bg from palettes.json by name")
    parser.add_argument("--input", default="frames", help="Directory of raw frame images")
    parser.add_argument("--output", default="preview_invert.mp4", help="Output video path")
    parser.add_argument("--cell-size", type=int, default=10, help="Grid cell size in pixels")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--palette", nargs="+", default=None, metavar="HEX",
                        help="Palette of hex colors to snap cells to. Omit to use original cell colors.")
    parser.add_argument("--bw", action="store_true",
                        help="Shortcut for --palette #000000 #ffffff")
    parser.add_argument("--grain", type=float, default=0, metavar="INTENSITY",
                        help="Film grain intensity (0=off, 25=subtle, 50=heavy)")
    parser.add_argument("--supersample", type=int, default=4, metavar="N",
                        help="Render at Nx resolution then downscale for smoother edges")
    parser.add_argument("--squares", type=int, default=50, metavar="PCT",
                        help="Percentage of cells rendered as squares instead of circles (0-100)")
    parser.add_argument("--bg", default="#ffffff", metavar="HEX", help="Background color as hex")
    parser.add_argument("--frame-stack", default=None, metavar="PATH",
                        help="Output path for a composite PNG of all frames stacked")
    args = parser.parse_args()

    if args.preset:
        preset = _load_preset(args.preset)
        if not args.palette:
            args.palette = preset.get("palette")
        if args.bg == "#ffffff" and "bg" in preset:
            args.bg = preset["bg"]

    if args.bw and not args.palette:
        args.palette = BW_PALETTE
    if args.palette:
        palette_rgb = [_hex_to_rgb(c) for c in args.palette]
        palette_bgr = np.array([(b, g, r) for (r, g, b) in palette_rgb], dtype=np.uint8)
    else:
        palette_bgr = None

    bg_rgb = _hex_to_rgb(args.bg)
    bg_bgr = (bg_rgb[2], bg_rgb[1], bg_rgb[0])

    paths = sorted(
        glob.glob(os.path.join(args.input, "*.png"))
        + glob.glob(os.path.join(args.input, "*.jpg"))
        + glob.glob(os.path.join(args.input, "*.jpeg"))
    )
    if not paths:
        print(f"No frames found in {args.input}/")
        return

    sample = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    grain_rng = np.random.RandomState(0) if args.grain > 0 else None
    stack = np.full((h, w, 3), bg_bgr, dtype=np.uint8) if args.frame_stack else None

    palette_label = args.palette if args.palette else "source colors"
    print(f"Processing {len(paths)} frames (cell_size={args.cell_size}, palette={palette_label})...")
    for i, path in enumerate(paths):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        rendered = render_invert(
            frame,
            args.cell_size,
            seed=i,
            palette_bgr=palette_bgr,
            supersample=args.supersample,
            squares_pct=args.squares,
            bg_bgr=bg_bgr,
        )
        if grain_rng is not None:
            rendered = apply_film_grain(rendered, grain_rng, intensity=args.grain)
        writer.write(rendered)

        if stack is not None:
            mask = np.any(rendered != bg_bgr, axis=2)
            stack[mask] = rendered[mask]

        if (i + 1) % 50 == 0 or i == len(paths) - 1:
            print(f"  {i + 1}/{len(paths)}")

    writer.release()
    print(f"Saved {args.output}")

    if stack is not None:
        cv2.imwrite(args.frame_stack, stack)
        print(f"Saved frame stack {args.frame_stack}")


if __name__ == "__main__":
    main()
