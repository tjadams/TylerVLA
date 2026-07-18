#!/usr/bin/env bash
# 
# Overview:
# Record SO-ARM-101 teleoperation episodes into a LeRobot dataset.
# Later, convert to TylerVLA .npz/.json. 

# Details:
# Records NUM_EPISODES back-to-back
# Each episode ≤ episode_time_s (default 60s)
# keys go to the --display_data / rerun.io window, not the terminal
# macOS: these keys use pynput and may need accessibility permissions
# See ../../architecture/lerobot_training.md for the normalization chain.
# 

# Usage to start recording:
# 1. conda activate lerobot
# 2. real_robot/recording_training_data/record_episode.sh

# Controls during recording: 
# Right arrow: end episode
# Left arrow: re-record current episode
# Escape: finish recording (encodes video, writes metadata)
# Ctrl+C: breaks recording and leaves partial files

# Output layout (per episode): 
# data/*/episode_*.parquet holds the numeric time-series
#   action [6] = leader 
#   observation.state [6] = follower
#   timestamps
# videos/*/episode_*.mp4
# Parquet + mp4 are written only at episode/session finalize
set -euo pipefail

# ---- Configure these ----
FOLLOWER_PORT="/dev/tty.usbmodem5A460830061"
LEADER_PORT="/dev/tty.usbmodem5A460825831"
CAMERA_INDEX=1                    # from probe_cameras.py (iPhone, not 0/FaceTime)
FPS=30                            
NUM_EPISODES=5
EPISODE_TIME_S=120               # per-episode cap, will end the episode if reached
RESET_TIME_S=10                  # pause between episodes to reset the environment
REPO_ID="tylervla/pick-place"
# TASK="pick up the ball and place it in the bowl"
TASK="pick up the medicine bottle and place it in the bowl"
# Timestamped so each run writes a fresh dataset — lerobot errors if the root already exists.
DATASET_ROOT="${HOME}/tylervla_datasets/pick-place_$(date +%Y%m%d_%H%M%S)"
# -------------------------

# push_to_hub defaults to true — keep it false to stay local, no HuggingFace upload.
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: 1920, height: 1080, fps: ${FPS}}}" \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.fps="${FPS}" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.push_to_hub=false \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.episode_time_s="${EPISODE_TIME_S}" \
    --dataset.reset_time_s="${RESET_TIME_S}" \
    --dataset.single_task="${TASK}"

echo
echo "Recording done (stored locally at ${DATASET_ROOT}, not uploaded)."
echo "Convert to TylerVLA format with:"
echo "  python real_robot/convert_lerobot.py --dataset ${DATASET_ROOT} --out demos/"
