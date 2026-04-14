# Plan: Training TylerVLA for Sim Pick-and-Place

## Context
User wants to train TylerVLA to pick up the ball and place it in the bowl, deployed entirely
in simulation. The question is: sim data vs real data, and what needs to be built.

---

## Answer: Use Simulation Data (Sim-to-Sim)

Since the deployment target is MuJoCo simulation (not a real robot), use sim data exclusively.
Collecting real robot data and training on it would introduce a sim-to-real gap going the *wrong*
direction (real-to-sim). Sim data is also faster to collect, reproducible, and doesn't require
a physical robot.

## Full Pipeline

```
1. Teleop in MuJoCo → .npz + .json 
2. model/train.py → model.pt
3. simulate.py (policy loop) with model.pt
```

### Step 1: Collect Demonstration Data (`simulation/collect_demos.py`)
A script that:
- Launches the MuJoCo scene (reuses `_load_scene_model`, `_place_robot_on_table`)
- Lets user teleoperate the robot via the viewer Controls panel (drag sliders)
- Records at ~20 Hz:
  - `images`: RGB frames from `gripper_cam` (128×128 uint8) — the same camera the policy will see
  - `joints`: `data.ctrl[:]` or `data.qpos[joint_indices]` (6 floats, float32)
- On exit: saves `demos/demo_001.npz` and `demos/demo_001.json`
- The `.json` repeats `"pick up the ball and place it in the bowl"` for every frame (constant command)
- Target: ~5–10 demonstrations of the full pick-and-place task (~30–60s each)

**Key design choices:**
- Record `data.ctrl` (actuator targets set by sliders) as the action — this is what the policy
  will predict and write back during inference
- Record from `gripper_cam` only (what the policy "sees") — not the overview camera
- Collect at 20 Hz (record every ~2nd frame at 100 Hz physics)

### Step 2: Train (`model/train.py`)

Already complete — no changes needed. Just run:
```bash
python -c "from model import train; train('demos/merged.npz', 'demos/merged.json', 'runs/pick_place_v1')"
```
`model/train.py` expects `.npz` files (images + joints) and `.json` files (texts) in the data dir.

### Step 3: Wire Inference into Simulation (`simulation/simulate.py`)

Implement `run_policy_and_actuate_robot` (currently a stub at line 225) to:
1. Render `gripper_cam` → preprocess to 128×128
2. Run policy forward pass → predicted normalized joint targets
3. Denormalize with `j_mean` / `j_std`
4. Write to `data.ctrl[:]` (let MuJoCo's position actuators handle the rest)
5. Apply optional exponential smoothing (already in inference.py)

Load policy once before the loop, pass into the loop.

## Verification (End-to-End)

1. Run `mjpython collect_demos.py` → teleop robot to complete pick-and-place → saves `demos/demo_001.npz`
2. Run training → loss converges → saves `runs/pick_place_v1/model.pt`
3. Run `mjpython simulate.py` → policy drives robot → robot picks up ball, places in bowl

End-to-end pipeline

  # 1. Collect 5-10 demos (teleop via viewer sliders)
  mjpython simulation/collect_demos.py

  # 2. Merge all demos
  mjpython simulation/collect_demos.py --merge

  # 3. Train
  python -c "from model import train; train('demos/merged.npz', 'demos/merged.json', 'runs/pick_place_v1')"

  # 4. Run policy in sim
  mjpython simulation/simulate.py --policy
