# Decision: Use LeRobot's recorder for real-world data collection

## Context
TylerVLA's `simulation/collect_demos.py` is MuJoCo-only — it renders a simulated gripper camera and records via viewer sliders. It cannot talk to real hardware. For real-robot data collection on the SO-ARM-101, we need a different approach.

Two options:
1. Write a custom data collection script using LeRobot's `SO101Follower` Python API directly
2. Use LeRobot's built-in `lerobot.record` CLI, then convert to TylerVLA's format

## Decision
Use LeRobot's `lerobot.record` + a conversion script.

## Why
- LeRobot already handles camera sync, motor reading, and leader-follower teleoperation — reimplementing this is unnecessary work
- Data is stored locally at `~/.cache/huggingface/lerobot/<repo_id>/` with no HuggingFace data upload required (just don't run `push_to_hub`)
- Joint values are already in the same normalized space `real_robot/inference.py` uses (`SO101FollowerConfig(use_degrees=False)` → -100..+100 for body joints, 0..100 for gripper)

## Data format: LeRobot vs TylerVLA

### LeRobot stores
```
~/.cache/huggingface/lerobot/<repo_id>/
├── data/chunk-000/episode_000000.parquet   # action [6], observation.state [6], timestamps
├── videos/chunk-000/observation.images.front/episode_000000.mp4  # 1920x1080 video
└── meta/
    ├── info.json          # fps, features schema, robot_type
    ├── tasks.jsonl        # {"task_index": 0, "task": "pick up the ball..."}
    └── episodes.jsonl     # {"episode_index": 0, "tasks": [...], "length": N}
```

- **Joints:** parquet columns `action` (leader commanded) and `observation.state` (follower actual), both float32 [6]
- **Images:** mp4 video, not raw frames
- **Text:** task string in `tasks.jsonl`

### TylerVLA expects
```
demos/merged.npz   # images uint8 [N, 128, 128, 3], joints float32 [N, 6]
demos/merged.json   # {"text": ["command", "command", ...]}  length N
```

### Conversion script needs to
1. Read parquet → extract `observation.state` as joints [N, 6]
2. Decode mp4 frames → resize to 128×128 → uint8 images [N, H, W, 3]
3. Read task text from `tasks.jsonl` → repeat N times
4. Save as `.npz` + `.json`

## Normalization chain
LeRobot normalized joints (-100..+100) → `model/train.py` further normalizes (subtract mean, divide by std) → model predicts in normalized space → `real_robot/inference.py` denormalizes (`pred * std + mean`) back to LeRobot space → `send_action()` accepts LeRobot normalized values.

No additional conversion needed between collection and inference.

## Recording command
From `SO-ARM-101/scripts/teleoperation/record-episode.sh`:
```bash
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

## Existing data
One episode already recorded: 1800 frames at 30 fps, task "Grab the blue stress ball", stored at `~/.cache/huggingface/lerobot/abc2/so-arm-101/`.
