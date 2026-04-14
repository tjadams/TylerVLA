# TylerVLA
From-scratch/vibe-coded implementation of the simplest possible VLA that me (and ChatGPT/Claude) could think of. 

## Motivation
The idea is to use very minimal training data (ideally 5-10 training episodes), simulate, and then deploy to a real SO-ARM-101 robot arm. Other models tried which were quite complex for a Robotics/deep-learning beginner, or required too much training data: OpenVLA, ACT.

## Usage - Real Robot (SO-ARM-101 via LeRobot)

Requires `conda activate lerobot` and the follower arm connected via USB.

```bash
# Test connection (reads joints + camera frame, then disconnects)
python test_robot_connection.py

# Run trained policy on the real arm at 10 Hz
python real_robot/inference.py
```

Hardware config (from SO-ARM-101 setup):
- Follower port: `/dev/tty.usbmodem5A460830061`
- Camera: OpenCV index 0

`real_robot/inference.py` connects to the arm via LeRobot's `SO101Follower`, runs the policy loop with exponential smoothing (alpha=0.2), and disconnects on exit.

## Data Collection & Training (real robot)

1. Record teleoperation demos via LeRobot (run as many times as needed):
```bash
conda activate lerobot
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460830061 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 5}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A460825831 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=tylervla/pick-place \
    --dataset.num_episodes=1 \
    --dataset.single_task="pick up the ball and place it in the bowl"
```
Data is stored locally at `~/.cache/huggingface/lerobot/<repo_id>/`.

2. Convert LeRobot dataset to TylerVLA format (one-time, after all demos collected):
```bash
python real_robot/convert_lerobot.py --dataset ~/.cache/huggingface/lerobot/tylervla/pick-place --out demos/
```
Produces `demos/merged.npz` + `demos/merged.json`. Re-run if you collect more demos.

3. Train:
```bash
conda activate pytorch
python -c "from model import train; train('demos/merged.npz', 'demos/merged.json', 'runs/pick_place_v1')"
```

## Usage - sim
1. conda activate pytorch
- Has robot_descriptions package
2. mjpython simulation/simulate.py

## Project Goals
(a) code VLA end-to-end from scratch / vibe-coded 
(b) train on literally a couple minutes of real-world data from my SO-ARM-101
(c) maybe simulate in MuJoCo
(d) deploy on an SO-ARM-101

## Overview
Tyler VLA = frozen encoders + tiny action head (BC)

Frozen V+L encoders → tiny policy head → actions
(You only train the tiny head; everything else is fixed.)

CLIP-BC Policy
- Vision: frozen CLIP image encoder (or any pretrained CNN/Vit you can load)
- Language: frozen CLIP text encoder (same model)
- Fusion (combine vision + language): concatenate embeddings (or FiLM if you want one extra step)
- Policy head: small MLP (2–4 layers)
- Action output: SO-ARM-101 commands (e.g., Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)

Another idea:
- Vision encoder: tiny CNN
- Text encoder: tiny tokenizer + Embedding + GRU
- Fusion: concat → MLP
- Output: joint positions (regression)

Why this fits “couple minutes”: you’re not trying to learn perception or language from scratch—just a small mapping from already-meaningful embeddings to actions.

Inputs
- Image: 128×128 or 224×224 RGB
- Text: command string (e.g., “pick up the red block”)

Forward pass
v = VisionEncoder(img) → 512 or 768 dim (frozen)
t = TextEncoder(text) → 512 or 768 dim (frozen)
z = concat([v, t]) (optionally add previous action / proprio)
â = MLP(z) → action dims

Loss
- Behavior cloning (supervised): L = MSE(â, a) (and BCE for gripper if binary)

Inference loop
- Read camera frame
- Encode text once per episode
- Run policy at e.g. 10–30 Hz
- Send action to SO-ARM-101 controller

Couple minutes of data can work, but only under these conditions:
- You’re training one task (or a couple very similar tasks)
- You can run at 10–30 Hz to get enough samples (2 min @ 20 Hz ≈ 2400 steps)
- You keep the policy output simple (delta pose + gripper)
- You do normalization + action smoothing
- You accept it may overfit and behave well only in the same setup/lighting

## Data format
Record demonstrations at (say) 10–30 Hz and save one dataset file:
- images: uint8 [N, H, W, 3] (e.g., 128×128)
- joints: float32 [N, J] (J = number of joints)
- text: list of N strings (same command repeated is fine)
- optional: episode_id to prevent mixing normalization across tasks (not required)
- Easiest: store as a .npz plus a .json for texts. Example:
- demo.npz contains images, joints
- demo_text.json contains text array length N


## Docker

Run the test suite in a container with all dependencies pre-installed (CPU-only PyTorch, MuJoCo, OpenCV, etc.):

```bash
docker build -t tylervla .
docker run tylervla
```

No conda or local setup required — everything is installed at build time. Note: `lerobot` is not installed in the Docker image. Real-robot tests and deployment require a local conda environment (`conda activate lerobot`).

Note: commit history for TylerVLA older than 4/14/26 is available in my learning-deeplearning repo, up to this commit: https://github.com/tjadams/Learning-DeepLearning/commit/b96fa9789b0eb16df23883ee463960204741e6db