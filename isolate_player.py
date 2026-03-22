"""
isolate_player.py — Click on objects to isolate them across all frames using SAM 2.

Click on the player, basketball, hoop, or anything else you want to keep.
Each click tracks a separate object. Close the window when done selecting.

Usage:
    python isolate_player.py [--frames-dir frames] [--output-dir isolated]
                             [--checkpoint checkpoints/sam2.1_hiera_large.pt]
                             [--preview]
"""

import argparse
import os
import sys
import warnings

import cv2
import numpy as np

# Matplotlib backend must be set before importing pyplot
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt

import torch


# ── Device selection ─────────────────────────────────────────────────────────

def select_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


# ── Click UI ─────────────────────────────────────────────────────────────────

def get_click_points(frames_dir: str, frame_names: list[str]) -> list[tuple[int, int, int]]:
    """Browse frames with arrow keys, click objects on any frame.

    Returns list of (x, y, frame_idx) tuples — each click becomes a
    separate SAM 2 object prompt on the frame where it was placed.

    Controls:
        Left/Right arrow  — previous/next frame
        Shift+Left/Right  — jump 10 frames
        Left click        — new object
        Right click       — reinforce selected object (same obj on a new frame)
        1-9 keys          — select existing object to reinforce
        Close window      — done selecting
    """
    state = {"idx": 0, "next_obj": 1, "active_obj": None}
    # Each click: (x, y, frame_idx, obj_id)
    clicks = []
    colors = ["red", "lime", "cyan", "magenta", "yellow", "orange"]

    fig, ax = plt.subplots(figsize=(12, 7))

    def obj_color(obj_id):
        return colors[(obj_id - 1) % len(colors)]

    def status_text():
        n_objects = len(set(c[3] for c in clicks))
        reinforce = f"  |  reinforce: obj {state['active_obj']}" if state["active_obj"] else ""
        return (
            f"Frame {state['idx']}/{len(frame_names)-1}  |  "
            f"{n_objects} object(s), {len(clicks)} click(s){reinforce}\n"
            f"L-click=new obj  |  R-click=reinforce  |  1-9=select obj  |  ←/→=navigate"
        )

    def draw_frame():
        ax.clear()
        path = os.path.join(frames_dir, frame_names[state["idx"]])
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        # Redraw clicks on this frame
        for cx, cy, fi, oid in clicks:
            if fi == state["idx"]:
                color = obj_color(oid)
                ax.plot(cx, cy, "+", color=color, markersize=15, markeredgewidth=2)
                ax.annotate(f" obj {oid}", (cx, cy),
                            color=color, fontsize=10, fontweight="bold")
        ax.set_title(status_text(), fontsize=10)
        ax.axis("off")
        fig.canvas.draw()

    def on_key(event):
        step = 10 if "shift" in (event.modifiers or set()) else 1
        if event.key in ("right", "shift+right"):
            state["idx"] = min(state["idx"] + step, len(frame_names) - 1)
            draw_frame()
        elif event.key in ("left", "shift+left"):
            state["idx"] = max(state["idx"] - step, 0)
            draw_frame()
        elif event.key in "123456789":
            obj_id = int(event.key)
            if obj_id < state["next_obj"]:
                state["active_obj"] = obj_id
                ax.set_title(status_text(), fontsize=10)
                fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = int(event.xdata), int(event.ydata)

        if event.button == 3 or (event.button == 1 and state["active_obj"]):
            # Right click or left click with active obj → reinforce existing object
            if state["active_obj"]:
                obj_id = state["active_obj"]
            elif clicks:
                obj_id = clicks[-1][3]  # reinforce most recent object
            else:
                return  # nothing to reinforce
        else:
            # Left click → new object
            obj_id = state["next_obj"]
            state["next_obj"] += 1
            state["active_obj"] = None

        clicks.append((x, y, state["idx"], obj_id))
        color = obj_color(obj_id)
        ax.plot(x, y, "+", color=color, markersize=15, markeredgewidth=2)
        ax.annotate(f" obj {obj_id}", (x, y),
                    color=color, fontsize=10, fontweight="bold")
        ax.set_title(status_text(), fontsize=10)
        fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    draw_frame()
    plt.tight_layout()
    plt.show()

    if not clicks:
        print("No clicks registered — exiting.")
        sys.exit(1)

    for x, y, fi, oid in clicks:
        print(f"  Object {oid}: ({x}, {y}) on frame {fi}")
    return clicks


# ── Mask edge softening ─────────────────────────────────────────────────────

def soft_mask(mask: np.ndarray, blur_radius: int = 5) -> np.ndarray:
    """Return a [0-255] alpha channel with slightly softened edges."""
    alpha = (mask.astype(np.float32) * 255)
    ksize = blur_radius * 2 + 1
    alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
    return np.clip(alpha, 0, 255).astype(np.uint8)


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Isolate a player using SAM 2")
    parser.add_argument("--frames-dir", default="frames", help="Directory of JPEG frames")
    parser.add_argument("--output-dir", default="isolated", help="Output directory for PNGs")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_large.pt",
                        help="Path to SAM 2.1 checkpoint")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM 2 model config (Hydra relative path)")
    parser.add_argument("--preview", action="store_true",
                        help="Generate preview_isolated.mp4 on white background")
    parser.add_argument("--cloud", action="store_true",
                        help="Offload SAM 2 inference to Modal cloud GPU")
    parser.add_argument("--prompts", default=None,
                        help="Path to prompts.json to reuse (skips click UI)")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(script_dir, args.frames_dir)
    output_dir = os.path.join(script_dir, args.output_dir)
    checkpoint = os.path.join(script_dir, args.checkpoint)

    # Clear old output before a new run
    if os.path.isdir(output_dir):
        for old_file in os.listdir(output_dir):
            if old_file.endswith(".png"):
                os.remove(os.path.join(output_dir, old_file))
    os.makedirs(output_dir, exist_ok=True)

    # List frames
    frame_names = sorted(
        f for f in os.listdir(frames_dir) if f.endswith(".jpg")
    )
    if not frame_names:
        print(f"No .jpg frames found in {frames_dir}")
        sys.exit(1)
    print(f"Found {len(frame_names)} frames in {frames_dir}")

    # Override device via env var if needed
    env_device = os.environ.get("SAM2_DEVICE")
    if env_device:
        device = torch.device(env_device)
        print(f"Using device override: {device}")
    else:
        device = select_device()

    # ── Get click points ─────────────────────────────────────────────────────
    from collections import defaultdict

    if args.prompts:
        # Load saved prompts instead of opening click UI
        import json
        prompts_path = os.path.join(script_dir, args.prompts) if not os.path.isabs(args.prompts) else args.prompts
        with open(prompts_path) as f:
            prompts_data = json.load(f)
        prompts = defaultdict(list)
        for p in prompts_data["prompts"]:
            prompts[(p["obj_id"], p["frame_idx"])] = [tuple(pt) for pt in p["points"]]
        n_objects = len(set(p["obj_id"] for p in prompts_data["prompts"]))
        n_clicks = sum(len(p["points"]) for p in prompts_data["prompts"])
        print(f"Loaded prompts from {prompts_path}")
        print(f"Tracking {n_objects} object(s) with {n_clicks} prompt(s)...")
    else:
        clicks = get_click_points(frames_dir, frame_names)
        prompts = defaultdict(list)  # (obj_id, frame_idx) → [(x, y), ...]
        for x, y, fi, oid in clicks:
            prompts[(oid, fi)].append((x, y))
        n_objects = len(set(c[3] for c in clicks))
        print(f"Tracking {n_objects} object(s) with {len(clicks)} prompt(s)...")

    if args.cloud:
        # ── Cloud inference via Modal ────────────────────────────────────────
        import json
        import subprocess

        # Save prompts to JSON
        prompts_json = {"prompts": []}
        for (obj_id, frame_idx), pts in sorted(prompts.items()):
            prompts_json["prompts"].append({
                "obj_id": obj_id,
                "frame_idx": frame_idx,
                "points": pts,
            })

        prompts_path = os.path.join(script_dir, "prompts.json")
        with open(prompts_path, "w") as f:
            json.dump(prompts_json, f, indent=2)
        print(f"Saved prompts to {prompts_path}")

        # Run Modal
        print("Launching Modal cloud inference...")
        result = subprocess.run(
            [
                sys.executable, "-m", "modal", "run", "cloud_infer.py",
                "--frames-dir", frames_dir,
                "--prompts", prompts_path,
                "--output-dir", output_dir,
            ],
            cwd=script_dir,
        )
        if result.returncode != 0:
            print("Modal inference failed.")
            sys.exit(1)

        print(f"✅ Cloud inference complete — results in {output_dir}/")
        video_segments = None  # Not available locally for preview

    else:
        # ── Local inference ──────────────────────────────────────────────────
        # ── Build SAM 2 video predictor ──────────────────────────────────────
        print("Loading SAM 2 model...")
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            config_file=args.config,
            ckpt_path=checkpoint,
            device=device,
        )

        # ── Run inference ────────────────────────────────────────────────────
        print("Initializing video state...")
        with torch.inference_mode():
            state = predictor.init_state(video_path=frames_dir)

            # Add prompts grouped by (obj_id, frame)
            for (obj_id, frame_idx), pts in sorted(prompts.items()):
                points = np.array(pts, dtype=np.float32)
                labels = np.ones(len(pts), dtype=np.int32)  # all foreground
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

            print(f"Propagating {n_objects} object(s) across {len(frame_names)} frames...")

            # Propagate and merge all object masks per frame
            video_segments = {}
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                # mask_logits shape: (num_objects, 1, H, W) — union all objects
                combined = torch.any(mask_logits > 0.0, dim=0).cpu().numpy().squeeze()
                video_segments[frame_idx] = combined

        # ── Save isolated frames ────────────────────────────────────────────
        print("Saving isolated frames...")
        for i, fname in enumerate(frame_names):
            frame_bgr = cv2.imread(os.path.join(frames_dir, fname))
            if i not in video_segments:
                # No mask for this frame — save transparent
                h, w = frame_bgr.shape[:2]
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
            else:
                mask = video_segments[i]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                alpha = soft_mask(mask)
                rgba = np.dstack([frame_rgb, alpha])

            out_name = f"player_{i:05d}.png"
            cv2.imwrite(
                os.path.join(output_dir, out_name),
                cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
            )

        print(f"✅ Saved {len(frame_names)} isolated PNGs to {output_dir}/")

    # ── Optional preview video ───────────────────────────────────────────────
    if args.preview:
        print("Generating preview video...")
        preview_path = os.path.join(script_dir, "preview_isolated.mp4")
        sample = cv2.imread(os.path.join(frames_dir, frame_names[0]))
        h, w = sample.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(preview_path, fourcc, 24.0, (w, h))

        if video_segments is not None:
            # Local inference — use in-memory masks
            for i, fname in enumerate(frame_names):
                canvas = np.full((h, w, 3), 255, dtype=np.uint8)
                if i in video_segments:
                    frame_bgr = cv2.imread(os.path.join(frames_dir, fname))
                    mask = video_segments[i]
                    alpha = soft_mask(mask).astype(np.float32) / 255.0
                    alpha_3ch = np.dstack([alpha, alpha, alpha])
                    canvas = (frame_bgr * alpha_3ch + canvas * (1 - alpha_3ch)).astype(np.uint8)
                writer.write(canvas)
        else:
            # Cloud inference — read PNGs from output dir
            png_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".png"))
            for png_name in png_files:
                rgba = cv2.imread(os.path.join(output_dir, png_name), cv2.IMREAD_UNCHANGED)
                canvas = np.full((h, w, 3), 255, dtype=np.uint8)
                if rgba is not None and rgba.shape[2] == 4:
                    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
                    bgr = rgba[:, :, :3]
                    alpha_3ch = np.dstack([alpha, alpha, alpha])
                    canvas = (bgr * alpha_3ch + canvas * (1 - alpha_3ch)).astype(np.uint8)
                writer.write(canvas)

        writer.release()
        print(f"✅ Preview saved to {preview_path}")


if __name__ == "__main__":
    main()
