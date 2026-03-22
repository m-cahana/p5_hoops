"""
pipeline.py — Orchestrate the full prep/isolate/mosaic pipeline for a named video.

Usage:
    python pipeline.py dunk1 --url 'https://...' --start 00:01:30 --duration 3
    python pipeline.py dunk1 --step isolate
    python pipeline.py dunk1 --step mosaic --cell-size 8 --color '#000000'
"""

import argparse
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")


def video_dir(name: str) -> str:
    return os.path.join(VIDEOS_DIR, name)


def run_prep(vdir: str, url: str | None, start: str, duration: str):
    raw_dir = os.path.join(vdir, "raw")
    frames_dir = os.path.join(vdir, "frames")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    cmd = [
        os.path.join(SCRIPT_DIR, "prep_video.sh"),
        "--raw-dir", raw_dir,
        "--frames-dir", frames_dir,
    ]
    if url:
        cmd.extend([url, start, duration])
    else:
        cmd.extend(["", start, duration])

    print(f"=== PREP: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_isolate(vdir: str, cloud: bool = False, prompts: str | None = None):
    frames_dir = os.path.join(vdir, "frames")
    output_dir = os.path.join(vdir, "isolated")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "isolate_player.py"),
        "--frames-dir", frames_dir,
        "--output-dir", output_dir,
    ]
    if cloud:
        cmd.append("--cloud")
    if prompts:
        cmd.extend(["--prompts", prompts])

    print(f"=== ISOLATE: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_mosaic(vdir: str, cell_size: int, fps: int, color: str | None, density: float):
    input_dir = os.path.join(vdir, "isolated")
    output_path = os.path.join(vdir, "preview_mosaic.mp4")
    stack_path = os.path.join(vdir, "frame_stack.png")

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "mosaic_preview.py"),
        "--input", input_dir,
        "--output", output_path,
        "--cell-size", str(cell_size),
        "--fps", str(fps),
        "--density", str(density),
        "--frame-stack", stack_path,
    ]
    if color:
        cmd.extend(["--color", color])

    print(f"=== MOSAIC: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full video pipeline for a named clip"
    )
    parser.add_argument("video_name", help="Name of the video (determines videos/{name}/ directory)")
    parser.add_argument("--step", default="all", choices=["prep", "isolate", "mosaic", "all"],
                        help="Run only one step (default: all)")

    # Prep args
    parser.add_argument("--url", default=None, help="YouTube URL to download")
    parser.add_argument("--start", default="00:00:00", help="Start time (HH:MM:SS)")
    parser.add_argument("--duration", default="3", help="Duration in seconds")

    # Isolate args
    parser.add_argument("--cloud", action="store_true", help="Use Modal cloud GPU for isolation")
    parser.add_argument("--prompts", default=None, help="Path to prompts.json to reuse (skips click UI)")

    # Mosaic args
    parser.add_argument("--cell-size", type=int, default=10, help="Mosaic cell size")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--color", default=None, help="Single hex color for mosaic")
    parser.add_argument("--density", type=float, default=0.15, help="Path density (fraction of occupied pixels to walk)")

    args = parser.parse_args()

    vdir = video_dir(args.video_name)
    os.makedirs(vdir, exist_ok=True)

    steps = ["prep", "isolate", "mosaic"] if args.step == "all" else [args.step]

    for step in steps:
        if step == "prep":
            run_prep(vdir, args.url, args.start, args.duration)
        elif step == "isolate":
            run_isolate(vdir, cloud=args.cloud, prompts=args.prompts)
        elif step == "mosaic":
            run_mosaic(vdir, args.cell_size, args.fps, args.color, args.density)

    print(f"\nDone! Output in {vdir}/")


if __name__ == "__main__":
    main()
