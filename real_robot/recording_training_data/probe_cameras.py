"""Probe OpenCV camera indices to find your Continuity Camera iPhone.

LeRobot's `record` grabs frames by OpenCV device index. On a Mac the built-in
FaceTime camera is usually index 0, so a tripod iPhone (via Continuity Camera)
lands on a higher index. This script opens each index, reports its resolution/fps,
and saves a snapshot so you can eyeball which one is the iPhone — then paste that
index into `--robot.cameras="{ front: {index_or_path: <N>, ...}}"`.

See also:
    - ../../notes/recording_training_data.md — setup context
    - ../../architecture/lerobot_training.md — full record command

Usage:
    conda activate lerobot   # or any env with opencv-python
    python real_robot/recording_training_data/probe_cameras.py
    python real_robot/recording_training_data/probe_cameras.py --max-index 8 --out /tmp/cam_probe
"""

import argparse
from pathlib import Path

import cv2


def probe(max_index: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    found = []

    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        ret, frame = cap.read()
        snapshot = None
        if ret and frame is not None:
            snapshot = out_dir / f"camera_{i}.jpg"
            cv2.imwrite(str(snapshot), frame)

        cap.release()
        found.append((i, width, height, fps, snapshot))
        snap_note = f" → {snapshot}" if snapshot else " (no frame captured)"
        print(f"  index {i}: {width}x{height} @ {fps:.0f} fps{snap_note}")

    print()
    if not found:
        print(f"No cameras found in indices 0..{max_index}.")
        print("If the iPhone is missing: confirm Continuity Camera is on and the")
        print("phone is unlocked/nearby on the same Apple ID.")
        return

    print(f"Found {len(found)} camera(s). Open the snapshots in {out_dir} to identify")
    print("the iPhone (highest-resolution, tripod viewpoint), then use its index in")
    print("the LeRobot record command's --robot.cameras block.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-index", type=int, default=5, help="Highest OpenCV index to probe (default 5)")
    parser.add_argument("--out", type=str, default="/tmp/cam_probe", help="Directory for snapshots")
    args = parser.parse_args()

    probe(args.max_index, Path(args.out))


if __name__ == "__main__":
    main()
