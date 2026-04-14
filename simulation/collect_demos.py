"""
Teleoperation data collection for TylerVLA.

Usage:
    mjpython simulation/collect_demos.py              # collect a new demo (teleop via viewer sliders)
    mjpython simulation/collect_demos.py --merge      # merge all demos/ into demos/merged.npz + merged.json

After collecting demos, train with:
    python -c "from model import train; train('demos/merged.npz', 'demos/merged.json', 'runs/pick_place_v1')"
"""

import argparse
import json
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

from simulation.simulate import _load_scene_model, _place_robot_on_table

DEMOS_DIR = "demos"
COMMAND = "pick up the ball and place it in the bowl"
IMG_H, IMG_W = 128, 128

# Physics: timestep=0.002s → 500 Hz. 5 steps/iter + sleep(0.01) = real-time at 100 iters/s.
# Record every 5 iters = 20 Hz.
PHYSICS_STEPS_PER_ITER = 5
ITERS_PER_RECORD = 5  # 100 iters/s / 5 = 20 Hz recording


def _next_demo_base():
    os.makedirs(DEMOS_DIR, exist_ok=True)
    existing = sorted(f for f in os.listdir(DEMOS_DIR) if f.startswith("demo_") and f.endswith(".npz"))
    idx = len(existing) + 1
    return os.path.join(DEMOS_DIR, f"demo_{idx:03d}")


def collect_demo():
    model = _load_scene_model()
    _place_robot_on_table(model)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    images = []
    joints = []
    iter_count = 0

    print("=" * 60)
    print("TELEOP DATA COLLECTION")
    print("  Task:", COMMAND)
    print("  Controls: drag sliders in the viewer Controls panel")
    print("  Close the viewer window when the demo is complete.")
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_forward(model, data)

        while viewer.is_running():
            for _ in range(PHYSICS_STEPS_PER_ITER):
                mujoco.mj_step(model, data)
            viewer.sync()
            iter_count += 1

            if iter_count % ITERS_PER_RECORD == 0:
                renderer.update_scene(data, camera="gripper_cam")
                img = renderer.render().copy()  # HWC uint8 RGB
                images.append(img)
                joints.append(data.ctrl[:].copy().astype(np.float32))

            time.sleep(0.01)  # real-time pacing

    del renderer

    if len(images) == 0:
        print("No frames recorded — demo not saved.")
        return

    base = _next_demo_base()
    npz_path = base + ".npz"
    json_path = base + ".json"

    np.savez(
        npz_path,
        images=np.stack(images, axis=0).astype(np.uint8),
        joints=np.stack(joints, axis=0),
    )
    with open(json_path, "w") as f:
        json.dump({"text": [COMMAND] * len(images)}, f)

    duration_s = len(images) / 20.0
    print(f"Saved {len(images)} frames ({duration_s:.1f}s @ 20 Hz) -> {npz_path}")
    print(f"Run --merge once you have enough demos, then train.")


def merge_demos():
    """Concatenate all demo_NNN.npz files into demos/merged.npz + demos/merged.json."""
    npz_files = sorted(
        f for f in os.listdir(DEMOS_DIR)
        if f.startswith("demo_") and f.endswith(".npz")
    )
    if not npz_files:
        print(f"No demo files found in {DEMOS_DIR}/")
        return

    all_images = []
    all_joints = []
    all_texts = []

    for npz_name in npz_files:
        base = npz_name[:-4]
        npz_path = os.path.join(DEMOS_DIR, npz_name)
        json_path = os.path.join(DEMOS_DIR, base + ".json")

        arr = np.load(npz_path)
        all_images.append(arr["images"])
        all_joints.append(arr["joints"])

        with open(json_path) as f:
            all_texts.extend(json.load(f)["text"])

        print(f"  {npz_name}: {len(arr['images'])} frames")

    merged_images = np.concatenate(all_images, axis=0)
    merged_joints = np.concatenate(all_joints, axis=0)

    out_npz = os.path.join(DEMOS_DIR, "merged.npz")
    out_json = os.path.join(DEMOS_DIR, "merged.json")
    np.savez(out_npz, images=merged_images, joints=merged_joints)
    with open(out_json, "w") as f:
        json.dump({"text": all_texts}, f)

    print(f"Merged {len(npz_files)} demos: {len(all_texts)} total frames -> {out_npz}")
    print(f"To train: python -c \"from model import train; train('{out_npz}', '{out_json}', 'runs/pick_place_v1')\"")


def main():
    parser = argparse.ArgumentParser(description="TylerVLA demo collector")
    parser.add_argument("--merge", action="store_true", help="Merge all demos/ into merged.npz/json")
    args = parser.parse_args()

    if args.merge:
        merge_demos()
    else:
        collect_demo()


if __name__ == "__main__":
    main()
