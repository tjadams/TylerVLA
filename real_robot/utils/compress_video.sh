#!/usr/bin/env bash
#
# compress_video.sh — shrink an mp4 to a target file size using two-pass H.264 on ffmpeg.
#
# Two-pass encoding is used because it lets ffmpeg hit a *specific* size:
# we compute a bitrate budget from (target size / duration), then let the
# first pass analyze the video and the second pass spend that budget wisely.
#
# Usage:
#   ./compress_video.sh input.mp4                 # default 9 MB target
#   ./compress_video.sh input.mp4 -s 9            # target 9 MB
#   ./compress_video.sh input.mp4 -s 25 -o out.mp4
#   ./compress_video.sh input.mp4 -a 96           # cap audio at 96 kbps
#
set -euo pipefail

TARGET_MB=9
AUDIO_KBPS=128
OUTPUT=""
INPUT=""

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--size)   TARGET_MB="$2"; shift 2 ;;
    -a|--audio)  AUDIO_KBPS="$2"; shift 2 ;;
    -o|--output) OUTPUT="$2"; shift 2 ;;
    -h|--help)   usage 0 ;;
    -*)          echo "Unknown option: $1" >&2; usage 1 ;;
    *)           INPUT="$1"; shift ;;
  esac
done

[[ -z "$INPUT" ]] && { echo "Error: no input file given." >&2; usage 1; }
[[ -f "$INPUT" ]] || { echo "Error: file not found: $INPUT" >&2; exit 1; }
# Resolve ffmpeg/ffprobe. Prefer the Homebrew build over whatever is first on
# PATH: an active conda env (e.g. lerobot) may shadow it with an older ffmpeg
# that lacks decoders like AV1 (libdav1d). Override with FFMPEG=/path env vars.
resolve_bin() {
  local name="$1" override="$2"
  if [[ -n "$override" ]]; then echo "$override"; return; fi
  if [[ -x "/opt/homebrew/bin/$name" ]]; then echo "/opt/homebrew/bin/$name"; return; fi
  if [[ -x "/usr/local/bin/$name" ]]; then echo "/usr/local/bin/$name"; return; fi
  command -v "$name" 2>/dev/null
}
FFMPEG=$(resolve_bin ffmpeg "${FFMPEG:-}")
FFPROBE=$(resolve_bin ffprobe "${FFPROBE:-}")
[[ -x "$FFMPEG"  ]] || { echo "Error: ffmpeg not found. Run: brew install ffmpeg" >&2; exit 1; }
[[ -x "$FFPROBE" ]] || { echo "Error: ffprobe not found. Run: brew install ffmpeg" >&2; exit 1; }

if [[ -z "$OUTPUT" ]]; then
  base="${INPUT%.*}"
  OUTPUT="${base}_compressed.mp4"
fi

# Duration in seconds (float).
DURATION=$("$FFPROBE" -v error -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 "$INPUT")

# Bitrate math (all in kbit/s):
#   total budget = target_MB * 8192 kbit / duration_s   (1 MB = 8192 kbit)
#   video budget = total budget - audio bitrate
TOTAL_KBPS=$(awk -v mb="$TARGET_MB" -v d="$DURATION" 'BEGIN { printf "%d", (mb*8192)/d }')
VIDEO_KBPS=$(awk -v t="$TOTAL_KBPS" -v a="$AUDIO_KBPS" 'BEGIN { printf "%d", t-a }')

if [[ "$VIDEO_KBPS" -lt 64 ]]; then
  echo "Error: target too small — video budget is ${VIDEO_KBPS} kbps (<64)." >&2
  echo "Raise --size or lower --audio." >&2
  exit 1
fi

IN_SIZE=$(du -h "$INPUT" | cut -f1)
echo "Input:        $INPUT ($IN_SIZE, ${DURATION%.*}s)"
echo "Target size:  ${TARGET_MB} MB"
echo "Video bitrate: ${VIDEO_KBPS} kbps | Audio: ${AUDIO_KBPS} kbps"
echo

# macOS mktemp needs a template; this works on both macOS and Linux.
PASSLOG=$(mktemp -t ffpass)

cleanup() { rm -f "${PASSLOG}"*.log "${PASSLOG}"*.log.mbtree "$PASSLOG"; }
trap cleanup EXIT

echo "Pass 1/2..."
"$FFMPEG" -y -i "$INPUT" -c:v libx264 -b:v "${VIDEO_KBPS}k" \
  -pass 1 -passlogfile "$PASSLOG" -an -f mp4 /dev/null

echo "Pass 2/2..."
"$FFMPEG" -y -i "$INPUT" -c:v libx264 -b:v "${VIDEO_KBPS}k" \
  -pass 2 -passlogfile "$PASSLOG" \
  -c:a aac -b:a "${AUDIO_KBPS}k" \
  -movflags +faststart "$OUTPUT"

OUT_SIZE=$(du -h "$OUTPUT" | cut -f1)
echo
echo "Done: $OUTPUT ($OUT_SIZE)"
