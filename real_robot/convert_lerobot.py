"""Convert LeRobot dataset to TylerVLA .npz + .json format.

LeRobot stores teleoperation data as parquet (joints) + mp4 (video). TylerVLA's
training pipeline expects .npz (images uint8 + joints float32) + .json (text commands).
This script bridges the two: decode video frames, resize to 128x128, extract joint
positions, and save in TylerVLA format. Run once after collecting all demos.

See also:
    - architecture/lerobot_training.md  — decision rationale + format comparison
    - README.md "Data Collection & Training" — end-to-end workflow

Usage:
    conda activate lerobot
    python real_robot/convert_lerobot.py --dataset ~/.cache/huggingface/lerobot/abc2/so-arm-101 --out demos/
    python real_robot/convert_lerobot.py --dataset ~/.cache/huggingface/lerobot/tylervla/pick-place --out demos/
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_SIZE = 128


def load_episodes(dataset_path: Path) -> list[dict]:
    """Read episodes.jsonl → list of episode metadata."""
    episodes = []
    with open(dataset_path / "meta" / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def load_tasks(dataset_path: Path) -> dict[int, str]:
    """Read tasks.jsonl → {task_index: task_string}."""
    tasks = {}
    with open(dataset_path / "meta" / "tasks.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            tasks[entry["task_index"]] = entry["task"]
    return tasks


def decode_video_frames(video_path: Path, expected_frames: int) -> np.ndarray:
    """Decode mp4 → uint8 [N, IMAGE_SIZE, IMAGE_SIZE, 3] RGB."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR → RGB, then resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(frame_resized)
    cap.release()

    if len(frames) != expected_frames:
        print(f"  Warning: video has {len(frames)} frames, parquet has {expected_frames}. Using min.")
    n = min(len(frames), expected_frames)
    return np.stack(frames[:n], axis=0).astype(np.uint8)


def convert_episode(dataset_path: Path, episode_idx: int, info: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert one episode → (images [N,H,W,3], joints [N,6], num_frames)."""
    chunk = episode_idx // info.get("chunks_size", 1000)

    # Read joints from parquet
    parquet_path = dataset_path / f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"
    df = pd.read_parquet(parquet_path)
    # Use observation.state (follower's actual position) — matches image frame timing
    joints = np.stack(df["observation.state"].values).astype(np.float32)

    # Decode video frames
    video_path = dataset_path / f"videos/chunk-{chunk:03d}/observation.images.front/episode_{episode_idx:06d}.mp4"
    images = decode_video_frames(video_path, expected_frames=len(joints))

    # Truncate to matching length
    n = min(len(images), len(joints))
    return images[:n], joints[:n], n


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to TylerVLA format")
    parser.add_argument("--dataset", type=str, required=True, help="Path to LeRobot dataset dir")
    parser.add_argument("--out", type=str, default="demos", help="Output directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_path / "meta" / "info.json") as f:
        info = json.load(f)

    episodes = load_episodes(dataset_path)
    tasks = load_tasks(dataset_path)

    print(f"Dataset: {dataset_path}")
    print(f"Episodes: {len(episodes)}, Tasks: {tasks}")

    all_images = []
    all_joints = []
    all_texts = []

    for ep in episodes:
        ep_idx = ep["episode_index"]
        task_text = ep["tasks"][0]
        print(f"  Converting episode {ep_idx}: \"{task_text}\" ({ep['length']} frames)")

        images, joints, n = convert_episode(dataset_path, ep_idx, info)
        all_images.append(images)
        all_joints.append(joints)
        all_texts.extend([task_text] * n)

    images_merged = np.concatenate(all_images, axis=0)
    joints_merged = np.concatenate(all_joints, axis=0)

    npz_path = out_dir / "merged.npz"
    json_path = out_dir / "merged.json"

    np.savez(npz_path, images=images_merged, joints=joints_merged)
    with open(json_path, "w") as f:
        json.dump({"text": all_texts}, f)

    print(f"\nSaved {len(all_texts)} frames across {len(episodes)} episodes")
    print(f"  images: {images_merged.shape} ({images_merged.dtype})")
    print(f"  joints: {joints_merged.shape} ({joints_merged.dtype})")
    print(f"  → {npz_path}")
    print(f"  → {json_path}")


if __name__ == "__main__":
    main()
