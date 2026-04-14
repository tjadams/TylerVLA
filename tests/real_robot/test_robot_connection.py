"""Smoke test: connect to SO-ARM-101 via LeRobot, read joints + camera, disconnect.

Usage:
    conda activate lerobot
    python test_robot_connection.py
"""

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

FOLLOWER_PORT = "/dev/tty.usbmodem5A460830061"


def main():
    config = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=0, fps=5, width=1920, height=1080,
            )
        },
    )
    robot = SO101Follower(config)

    print("Connecting...")
    robot.connect(calibrate=False)
    print("Connected.")

    obs = robot.get_observation()
    joints = {k: v for k, v in obs.items() if k.endswith(".pos")}
    print(f"Joint positions: {joints}")
    print(f"Camera frame shape: {obs['front'].shape}, dtype: {obs['front'].dtype}")

    robot.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
