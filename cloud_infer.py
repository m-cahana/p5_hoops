"""
cloud_infer.py — Modal app for SAM 2 video segmentation on a cloud GPU.

Receives click prompts (JSON) + video frames, runs SAM 2 propagation on an A10G,
and saves isolated PNGs back to a Modal Volume for download.

Usage:
    modal run cloud_infer.py --frames-dir frames --prompts prompts.json --output-dir isolated
"""

import modal
import os

app = modal.App("sam2-segmenter")

# ── Modal image with SAM 2 + dependencies ────────────────────────────────────
sam2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python-headless",
        "numpy",
    )
    .pip_install("git+https://github.com/facebookresearch/sam2.git")
)

# Persistent volume for model weights (survives across runs)
weights_volume = modal.Volume.from_name("sam2-model-weights", create_if_missing=True)
# Workspace volume for frames + results
workspace_volume = modal.Volume.from_name("sam2-workspace", create_if_missing=True)

WEIGHTS_DIR = "/weights"
WORKSPACE_DIR = "/workspace"
CHECKPOINT_NAME = "sam2.1_hiera_large.pt"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"


@app.cls(
    image=sam2_image,
    gpu="A10G",
    volumes={WEIGHTS_DIR: weights_volume, WORKSPACE_DIR: workspace_volume},
    timeout=600,
)
class SAM2Segmenter:
    @modal.enter()
    def load_model(self):
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        ckpt_path = os.path.join(WEIGHTS_DIR, CHECKPOINT_NAME)

        # Download checkpoint if not cached in volume
        if not os.path.exists(ckpt_path):
            import urllib.request
            print(f"Downloading SAM 2 checkpoint to volume...")
            urllib.request.urlretrieve(CHECKPOINT_URL, ckpt_path)
            weights_volume.commit()
            print("Checkpoint downloaded and cached.")

        print("Building SAM 2 video predictor...")
        self.predictor = build_sam2_video_predictor(
            config_file=CONFIG_NAME,
            ckpt_path=ckpt_path,
            device=torch.device("cuda"),
        )
        print("Model loaded on GPU.")

    @modal.method()
    def segment(self, job_id: str):
        """Run SAM 2 propagation using prompts and frames from the workspace volume."""
        import json
        import numpy as np
        import cv2
        import torch

        job_dir = os.path.join(WORKSPACE_DIR, job_id)
        frames_dir = os.path.join(job_dir, "frames")
        prompts_path = os.path.join(job_dir, "prompts.json")
        output_dir = os.path.join(job_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Reload volume to see uploaded files
        workspace_volume.reload()

        with open(prompts_path) as f:
            prompts_data = json.load(f)

        frame_names = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        print(f"Found {len(frame_names)} frames, {len(prompts_data['prompts'])} prompt group(s)")

        with torch.inference_mode():
            state = self.predictor.init_state(video_path=frames_dir)

            # Add prompts
            for prompt in prompts_data["prompts"]:
                obj_id = prompt["obj_id"]
                frame_idx = prompt["frame_idx"]
                points = np.array(prompt["points"], dtype=np.float32)
                labels = np.ones(len(prompt["points"]), dtype=np.int32)
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

            print("Propagating masks...")
            video_segments = {}
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                combined = torch.any(mask_logits > 0.0, dim=0).cpu().numpy().squeeze()
                video_segments[frame_idx] = combined

        # Save isolated PNGs
        print("Saving isolated frames...")
        for i, fname in enumerate(frame_names):
            frame_bgr = cv2.imread(os.path.join(frames_dir, fname))
            if i not in video_segments:
                h, w = frame_bgr.shape[:2]
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
            else:
                mask = video_segments[i]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                alpha = _soft_mask(mask)
                rgba = np.dstack([frame_rgb, alpha])

            out_name = f"player_{i:05d}.png"
            cv2.imwrite(
                os.path.join(output_dir, out_name),
                cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
            )

        workspace_volume.commit()
        print(f"Done — saved {len(frame_names)} PNGs to volume at {output_dir}")
        return len(frame_names)


def _soft_mask(mask, blur_radius: int = 5):
    """Return a [0-255] alpha channel with slightly softened edges."""
    import cv2
    import numpy as np
    alpha = mask.astype(np.float32) * 255
    ksize = blur_radius * 2 + 1
    alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
    return np.clip(alpha, 0, 255).astype(np.uint8)


# ── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    frames_dir: str = "frames",
    prompts: str = "prompts.json",
    output_dir: str = "isolated",
):
    import json
    import uuid

    frames_dir = os.path.abspath(frames_dir)
    prompts_path = os.path.abspath(prompts)
    output_dir = os.path.abspath(output_dir)

    # Validate inputs
    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Frames directory not found: {frames_dir}")
    if not os.path.isfile(prompts_path):
        raise SystemExit(f"Prompts file not found: {prompts_path}")

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if not frame_files:
        raise SystemExit(f"No .jpg frames found in {frames_dir}")

    with open(prompts_path) as f:
        prompts_data = json.load(f)

    job_id = uuid.uuid4().hex[:8]
    print(f"Job ID: {job_id}")
    print(f"Uploading {len(frame_files)} frames + prompts to Modal volume...")

    # Upload frames to workspace volume
    job_prefix = f"{job_id}/frames"
    with workspace_volume.batch_upload() as batch:
        for fname in frame_files:
            batch.put_file(os.path.join(frames_dir, fname), f"{job_prefix}/{fname}")

    # Upload prompts
    with workspace_volume.batch_upload() as batch:
        batch.put_file(prompts_path, f"{job_id}/prompts.json")

    print("Upload complete. Starting GPU inference...")

    # Run inference on GPU
    segmenter = SAM2Segmenter()
    n_frames = segmenter.segment.remote(job_id)

    print(f"Inference complete. Downloading {n_frames} PNGs...")

    # Download results
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = f"{job_id}/output/"

    for entry in workspace_volume.listdir(output_prefix):
        fname = os.path.basename(entry.path)
        if fname.endswith(".png"):
            local_path = os.path.join(output_dir, fname)
            with open(local_path, "wb") as out_f:
                for chunk in workspace_volume.read_file(entry.path):
                    out_f.write(chunk)

    print(f"Downloaded {n_frames} PNGs to {output_dir}/")

    # Clean up workspace volume
    workspace_volume.remove_file(job_id, recursive=True)
    print("Cleaned up remote workspace.")
