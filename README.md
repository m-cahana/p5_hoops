# p5 Hoops

Turn basketball highlight clips into mosaic-style animations.

## Pipeline

The full pipeline has three steps: **prep**, **isolate**, and **mosaic**. You can run them all at once or individually.

### 1. Prep — Download and extract frames

```bash
venv/bin/python pipeline.py shai2 --step prep --url 'https://youtube.com/...' --start 00:01:30 --duration 3
```

Downloads the clip with yt-dlp, trims it, and extracts frames to `videos/shai2/frames/`.

If you already have `videos/shai2/raw/source.mp4`, omit `--url`.

### 2. Isolate — Segment the player

```bash
venv/bin/python pipeline.py shai2 --step isolate
```

Opens a click UI on the first frame — click on the player (and ball, hoop, etc.) to select objects. SAM 2 tracks them across all frames and outputs RGBA PNGs to `videos/shai2/isolated/`.

Options:

- `--cloud` — run on Modal cloud GPU (recommended)
- `--prompts path/to/prompts.json` — reuse saved prompts (skips click UI)

### 3. Mosaic — Generate preview video

```bash
venv/bin/python pipeline.py shai2 --step mosaic --cell-size 10
```

Renders isolated frames as a grid of circles/squares and outputs `videos/shai2/preview_mosaic.mp4` and `frame_stack.png`.

Options:

- `--cell-size N` — grid cell size in pixels (default 10)
- `--squares PCT` — percentage of cells as squares instead of circles (0-100, default 0)
- `--color '#FF0000'` — single color for all cells (default: palette)
- `--grain 25` — film grain overlay (0=off)
- `--fps 30` — output framerate

### Run all steps at once

```bash
venv/bin/python pipeline.py shai2 --url 'https://youtube.com/...' --start 00:01:30 --duration 3 --cell-size 8
```
