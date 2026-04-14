# TylerVLA

Minimal Vision-Language-Action (VLA) model built from scratch. Architecture: frozen tiny CNN + GRU encoders feeding a trainable MLP policy head, trained via behavior cloning on MuJoCo teleoperation demos. Target hardware: SO-ARM-101 robot arm. 5–10 demo episodes are sufficient to train.

## Environment

```bash
conda activate pytorch
```

Use `mjpython` (not plain `python`) for any script that imports `mujoco`.

## Common Commands

```bash
# Collect a single teleoperation demo
mjpython simulation/collect_demos.py

# Merge all demos/ into demos/merged.npz + demos/merged.json
mjpython simulation/collect_demos.py --merge

# Train policy
python -c "from model import train; train('demos/merged.npz', 'demos/merged.json', 'runs/pick_place_v1')"

# Manual simulation
mjpython simulation/simulate.py

# Run trained policy in sim
mjpython simulation/simulate.py --policy
```

## Architecture (`model/model.py`)

- `SimpleTokenizer` — whitespace vocab tokenizer
- `TinyVisionEncoder` — tiny CNN (3→32→64→128 channels), output 128D
- `TinyTextEncoder` — Embedding + GRU, output 128D
- `TylerVLAPolicy` — concat(vision_embed, text_embed) → MLP → num_joints

Only the MLP policy head trains; encoders are frozen.

## File Roles

| File | Purpose |
|------|---------|
| `model/model.py` | Model classes + `DemoDataset` |
| `model/train.py` | Training loop; saves `runs/*/model.pt`, `tokenizer.json`, `joint_norm.npz` |
| `model_utils/policy_loader.py` | `load_policy` + `preprocess_image` — shared by sim and real robot |
| `simulation/collect_demos.py` | MuJoCo teleoperation data collection |
| `simulation/simulate.py` | Simulation environment (pick-and-place, ball→bowl); configure via constants at top |
| `real_robot/inference.py` | Real-robot deployment (SO-ARM-101 via LeRobot) |
| `real_robot/convert_lerobot.py` | Convert LeRobot datasets to TylerVLA `.npz`/`.json` format |
| `tests/` | Unit and smoke tests |

## Data Format

- `.npz`: `images` uint8 [N, 128, 128, 3], `joints` float32 [N, 6]
- `.json`: text command list, length N

## Key Constants in `simulation/simulate.py`

`POLICY_RUN_DIR`, `POLICY_COMMAND`, `POLICY_ALPHA`

## Device

Auto-detects CUDA → MPS → CPU.

## Testing

```bash
python -m pytest tests/
```

Validation loss is logged during training. Smoke test: `mjpython simulation/simulate.py --policy`.

## Docker

Runs the test suite in a containerized environment with all deps pre-installed (CPU-only PyTorch).

```bash
docker build -t tylervla .
docker run tylervla
```

Note: `lerobot` is not installed in the Docker image. Real-robot tests and `real_robot/inference.py` require a local conda environment (`conda activate lerobot`).
