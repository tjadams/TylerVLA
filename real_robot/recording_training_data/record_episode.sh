#!/usr/bin/env bash
# Record SO-ARM-101 teleoperation episodes into a LeRobot dataset, then convert
# to TylerVLA .npz/.json. Fill in the ports (ls /dev/tty.usbmodem*) and the
# iPhone camera index from probe_cameras.py before running.
#
#   conda activate lerobot
#   real_robot/recording_training_data/record_episode.sh
#
# See ../../architecture/lerobot_training.md for the normalization chain.
set -euo pipefail

# ---- Configure these ----
FOLLOWER_PORT="/dev/tty.usbmodem5A460830061"
LEADER_PORT="/dev/tty.usbmodem5A460825831"
CAMERA_INDEX=1                    # from probe_cameras.py (iPhone, not 0/FaceTime)
FPS=30                            # match existing episodes (abc2/so-arm-101 is 30 fps)
NUM_EPISODES=5
REPO_ID="tylervla/pick-place"
TASK="pick up the ball and place it in the bowl"
DATASET_ROOT="${HOME}/tylervla_datasets/pick-place"   # local folder to store the dataset
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
    --dataset.single_task="${TASK}"

echo
echo "Recording done (stored locally at ${DATASET_ROOT}, not uploaded)."
echo "Convert to TylerVLA format with:"
echo "  python real_robot/convert_lerobot.py --dataset ${DATASET_ROOT} --out demos/"
