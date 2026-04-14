"""Unit tests for convert_lerobot.py."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real_robot.convert_lerobot import convert_episode, decode_video_frames, load_episodes, load_tasks

NUM_FRAMES = 10
NUM_JOINTS = 6
IMG_H, IMG_W = 480, 640


class TestConvertLerobot(unittest.TestCase):
    def setUp(self):
        """Create a minimal fake LeRobot dataset on disk."""
        self.tmpdir = Path(tempfile.mkdtemp())
        self.dataset_path = self.tmpdir / "test-dataset"

        # Dirs
        (self.dataset_path / "meta").mkdir(parents=True)
        (self.dataset_path / "data" / "chunk-000").mkdir(parents=True)
        (self.dataset_path / "videos" / "chunk-000" / "observation.images.front").mkdir(parents=True)

        # meta/info.json
        info = {
            "codebase_version": "v2.1",
            "robot_type": "so101_follower",
            "total_episodes": 1,
            "total_frames": NUM_FRAMES,
            "fps": 30,
            "chunks_size": 1000,
        }
        with open(self.dataset_path / "meta" / "info.json", "w") as f:
            json.dump(info, f)

        # meta/tasks.jsonl
        with open(self.dataset_path / "meta" / "tasks.jsonl", "w") as f:
            f.write(json.dumps({"task_index": 0, "task": "pick up the ball"}) + "\n")

        # meta/episodes.jsonl
        with open(self.dataset_path / "meta" / "episodes.jsonl", "w") as f:
            f.write(json.dumps({"episode_index": 0, "tasks": ["pick up the ball"], "length": NUM_FRAMES}) + "\n")

        # data/chunk-000/episode_000000.parquet
        states = [np.random.uniform(-100, 100, NUM_JOINTS).astype(np.float32) for _ in range(NUM_FRAMES)]
        actions = [np.random.uniform(-100, 100, NUM_JOINTS).astype(np.float32) for _ in range(NUM_FRAMES)]
        df = pd.DataFrame({
            "observation.state": states,
            "action": actions,
            "timestamp": np.arange(NUM_FRAMES, dtype=np.float32) / 30.0,
            "frame_index": np.arange(NUM_FRAMES),
            "episode_index": np.zeros(NUM_FRAMES, dtype=np.int64),
            "index": np.arange(NUM_FRAMES),
            "task_index": np.zeros(NUM_FRAMES, dtype=np.int64),
        })
        df.to_parquet(self.dataset_path / "data" / "chunk-000" / "episode_000000.parquet")

        # videos/chunk-000/observation.images.front/episode_000000.mp4
        video_path = self.dataset_path / "videos" / "chunk-000" / "observation.images.front" / "episode_000000.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30, (IMG_W, IMG_H))
        for i in range(NUM_FRAMES):
            frame = np.full((IMG_H, IMG_W, 3), i * 25, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        self.states = states

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_tasks(self):
        tasks = load_tasks(self.dataset_path)
        self.assertEqual(tasks, {0: "pick up the ball"})

    def test_load_episodes(self):
        episodes = load_episodes(self.dataset_path)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["episode_index"], 0)
        self.assertEqual(episodes[0]["length"], NUM_FRAMES)

    def test_decode_video_frames(self):
        video_path = self.dataset_path / "videos" / "chunk-000" / "observation.images.front" / "episode_000000.mp4"
        frames = decode_video_frames(video_path, expected_frames=NUM_FRAMES)
        self.assertEqual(frames.shape, (NUM_FRAMES, 128, 128, 3))
        self.assertEqual(frames.dtype, np.uint8)

    def test_convert_episode(self):
        with open(self.dataset_path / "meta" / "info.json") as f:
            info = json.load(f)
        images, joints, n = convert_episode(self.dataset_path, 0, info)

        self.assertEqual(n, NUM_FRAMES)
        self.assertEqual(images.shape, (NUM_FRAMES, 128, 128, 3))
        self.assertEqual(joints.shape, (NUM_FRAMES, NUM_JOINTS))
        self.assertEqual(joints.dtype, np.float32)

        # Verify joints match observation.state
        for i in range(NUM_FRAMES):
            np.testing.assert_array_almost_equal(joints[i], self.states[i])


if __name__ == "__main__":
    unittest.main()
