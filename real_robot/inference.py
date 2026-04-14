"""Real-robot deployment for TylerVLA on SO-ARM-101 via LeRobot.

Connects to the follower arm, runs the trained policy in a loop at 10 Hz,
and sends joint position commands with exponential smoothing.

Usage:
    conda activate lerobot
    python real_robot/inference.py
"""

import time

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

from model_utils import load_policy, preprocess_image

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_robot: SO101Follower | None = None


# ---- SO-ARM-101 via LeRobot ----
def init_robot(port: str = "/dev/tty.usbmodem5A460830061", camera_index: int = 0):
    global _robot
    config = SO101FollowerConfig(
        port=port,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=camera_index, fps=5, width=1920, height=1080,
            )
        },
    )
    _robot = SO101Follower(config)
    _robot.connect(calibrate=False)


def disconnect_robot():
    global _robot
    if _robot is not None and _robot.is_connected:
        _robot.disconnect()
        _robot = None


def get_rgb_frame() -> np.ndarray:
    """Return an HWC uint8 RGB frame from the camera."""
    return _robot.cameras["front"].async_read()


def get_current_joint_positions() -> np.ndarray:
    """Return current joint positions [6] float32 in LeRobot normalized space."""
    pos = _robot.bus.sync_read("Present_Position")
    return np.array([pos[name] for name in JOINT_NAMES], dtype=np.float32)


def set_joint_positions(q_des: np.ndarray):
    """Send desired joint positions [6] to SO-ARM-101."""
    action = {f"{name}.pos": float(q_des[i]) for i, name in enumerate(JOINT_NAMES)}
    _robot.send_action(action)
# --------------------------------


def main(run_dir: str, command: str, hz: float = 10.0):
    model, tokenizer, j_mean, j_std, device = load_policy(run_dir)
    dt = 1.0 / hz

    init_robot()

    # Cache text tokens once (command usually constant per episode)
    text_ids = tokenizer.encode(command, max_len=16).unsqueeze(0).to(device)

    # Simple smoothing to reduce jitter
    alpha = 0.2  # 0..1; higher = more responsive, lower = smoother
    q_prev = None

    try:
        while True:
            t0 = time.time()

            img = get_rgb_frame()
            img_t = preprocess_image(img, image_size=128).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(img_t, text_ids).squeeze(0).cpu().numpy()  # [J] normalized
            q_des = pred_norm * j_std + j_mean  # de-normalize to LeRobot space

            # Safety / smoothing
            if q_prev is None:
                q_prev = get_current_joint_positions().astype(np.float32)
            q_cmd = (1 - alpha) * q_prev + alpha * q_des
            q_prev = q_cmd

            # Optional: clamp to joint limits if you have them
            # q_cmd = np.clip(q_cmd, q_min, q_max)

            set_joint_positions(q_cmd)

            # Rate control
            elapsed = time.time() - t0
            sleep_t = max(0.0, dt - elapsed)
            time.sleep(sleep_t)
    finally:
        disconnect_robot()


if __name__ == "__main__":
    main("runs/pick_place_v1", command="pick up the ball and place it in the bowl", hz=10.0)