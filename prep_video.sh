#!/usr/bin/env bash
# prep_video.sh — Download (optional), trim, and extract frames from highlight footage
# Usage: ./prep_video.sh [YOUTUBE_URL] [START_TIME] [DURATION_SECONDS]
#   YOUTUBE_URL   — if omitted, expects raw/source.mp4 to already exist
#   START_TIME    — HH:MM:SS offset into video (default: 00:00:00)
#   DURATION      — seconds to keep (default: 3)

set -euo pipefail

URL="${1:-}"
START="${2:-00:00:00}"
DURATION="${3:-3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/raw"
FRAMES_DIR="$SCRIPT_DIR/frames"

mkdir -p "$RAW_DIR" "$FRAMES_DIR"

# ── Step 1: Download if URL provided ─────────────────────────────────────────
if [[ -n "$URL" ]]; then
    echo "⬇  Downloading video..."
    yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
        --merge-output-format mp4 \
        -o "$RAW_DIR/source.mp4" \
        "$URL"
else
    if [[ ! -f "$RAW_DIR/source.mp4" ]]; then
        echo "Error: No URL provided and raw/source.mp4 not found." >&2
        exit 1
    fi
    echo "Using existing raw/source.mp4"
fi

# ── Step 2: Trim ─────────────────────────────────────────────────────────────
echo "✂  Trimming ${DURATION}s starting at ${START}..."
ffmpeg -y -ss "$START" -i "$RAW_DIR/source.mp4" -t "$DURATION" \
    -c:v libx264 -an "$RAW_DIR/trimmed.mp4"

# ── Step 3: Extract frames (5-digit zero-padded JPEG — SAM 2 requirement) ───
rm -f "$FRAMES_DIR"/*.jpg
echo "🎞  Extracting frames to frames/%05d.jpg..."
ffmpeg -y -i "$RAW_DIR/trimmed.mp4" "$FRAMES_DIR/%05d.jpg"

FRAME_COUNT=$(ls -1 "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "✅  Done — ${FRAME_COUNT} frames extracted to frames/"
