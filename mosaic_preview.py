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
   "#b80900", 
   '#0746a6',
]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _build_palette(single_color: str | None) -> list[tuple[int, int, int]]:
    if single_color:
        return [_hex_to_rgb(single_color)]
    return [_hex_to_rgb(h) for h in PALETTE_HEX]


def render_mosaic(image_rgba: np.ndarray, cell_size: int, seed: int, palette: list, supersample: int = 1, squares_pct: int = 0) -> np.ndarray:
    """Render occupied cells as filled circles (and optionally squares) with random palette colors.

    supersample: render at Nx resolution then downscale for smoother shapes.
    squares_pct: 0-100, percentage of cells rendered as squares instead of circles.
    """
    h, w = image_rgba.shape[:2]
    s = supersample
    canvas = np.full((h * s, w * s, 3), 255, dtype=np.uint8)

    alpha = image_rgba[:, :, 3]
    rng = np.random.RandomState(seed)

    for row in range(0, h, cell_size):
        for col in range(0, w, cell_size):
            cell_alpha = alpha[row : row + cell_size, col : col + cell_size]
            if cell_alpha.mean() < 30:
                continue

            chosen = palette[rng.randint(len(palette))]
            color_bgr = (chosen[2], chosen[1], chosen[0])

            cx = (col + cell_size // 2) * s
            cy = (row + cell_size // 2) * s
            radius = int(cell_size * 0.3 * s)

            if squares_pct > 0 and rng.randint(100) < squares_pct:
                half = radius
                pt1 = (cx - half, cy - half)
                pt2 = (cx + half, cy + half)
                cv2.rectangle(canvas, pt1, pt2, color_bgr, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (cx, cy), radius, color_bgr, -1, cv2.LINE_AA)

    if s > 1:
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return canvas


def apply_film_grain(frame: np.ndarray, rng: np.random.RandomState, intensity: float = 25.0) -> np.ndarray:
    """Overlay monochromatic film grain noise onto a BGR frame."""
    h, w = frame.shape[:2]
    noise = rng.normal(0, intensity, (h, w)).astype(np.float32)
    # Apply same noise to all channels (monochromatic grain)
    result = frame.astype(np.float32)
    for c in range(3):
        result[:, :, c] += noise
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Geometric mosaic preview from isolated PNGs")
    parser.add_argument("--input", default="isolated", help="Directory of RGBA PNGs")
    parser.add_argument("--output", default="preview_mosaic.mp4", help="Output video path")
    parser.add_argument("--cell-size", type=int, default=10, help="Grid cell size in pixels")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--color", default=None, metavar="HEX",
                        help="Single hex color for all cells (e.g. #000000). Omit to use full palette.")
    parser.add_argument("--grain", type=float, default=0, metavar="INTENSITY",
                        help="Film grain intensity (0=off, 25=subtle, 50=heavy)")
    parser.add_argument("--supersample", type=int, default=4, metavar="N",
                        help="Render circles at Nx resolution then downscale for smoother edges (default 4)")
    parser.add_argument("--squares", type=int, default=50, metavar="PCT",
                        help="Percentage of cells rendered as squares instead of circles (0-100, default 0)")
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
    grain_rng = np.random.RandomState(0) if args.grain > 0 else None
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

        mosaic = render_mosaic(frame, args.cell_size, seed=i, palette=palette, supersample=args.supersample, squares_pct=args.squares)
        if grain_rng is not None:
            mosaic = apply_film_grain(mosaic, grain_rng, intensity=args.grain)
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
