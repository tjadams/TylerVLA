"""MuJoCo simulation environment for TylerVLA.

Builds a tabletop pick-and-place scene (ball + bowl) with the SO-ARM-101 robot.
Supports manual viewing and running a trained policy in the loop.

Usage:
    mjpython simulation/simulate.py            # manual viewer
    mjpython simulation/simulate.py --policy   # run trained policy in sim
"""

import argparse
import os
import tempfile
import textwrap
import time
import xml.etree.ElementTree as ET
import multiprocessing as mp
import numpy as np
import mujoco
import mujoco.viewer
import torch

from robot_descriptions import so_arm101_mj_description

from model_utils import load_policy, preprocess_image

# Set to False to hide the cyan/orange camera position markers in the viewer
SHOW_CAMERA_MARKERS = False

# Policy settings (used when --policy flag is passed)
POLICY_RUN_DIR = "runs/pick_place_v1"
POLICY_COMMAND = "pick up the ball and place it in the bowl"
POLICY_ALPHA = 0.2  # exponential smoothing: 0=frozen, 1=no smoothing


def _get_simulated_camera_img(model, data, width=256, height=256):
  """
  Minimal offscreen render to RGB.
  For a real pipeline, you can use a wrist camera model or attach a MuJoCo camera.
  """
  renderer = mujoco.Renderer(model, height=height, width=width)
  renderer.update_scene(data, camera=None)  # or camera="your_cam_name"
  img = renderer.render()  # HWC uint8 RGB
  return img


# Scene simulating pick and place set up I have in my apartment
def _build_scene_xml(robot_xml_name: str) -> str:
  overview_marker_xml = """
        <!-- Cyan marker at overview camera position (0, 0, 1.5) -->
        <site name="overview_cam_marker" pos="0 0 1.5" size="0.04" material="cam_marker_overview" type="sphere"/>
        <!-- Stem pointing down to show look direction -->
        <site name="overview_cam_dir" pos="0 0 1.4" size="0.008 0.1" material="cam_marker_overview" type="cylinder"/>""" if SHOW_CAMERA_MARKERS else ""

  cam_marker_materials = """
        <material name="cam_marker_overview" rgba="0 1 1 0.9"    emission="1" reflectance="0"/>
        <material name="cam_marker_gripper"  rgba="1 0.5 0 0.9"  emission="1" reflectance="0"/>""" if SHOW_CAMERA_MARKERS else ""

  return textwrap.dedent(f"""\
    <mujoco model="tabletop_scene">
      <option timestep="0.002" gravity="0 0 -9.81"/>

      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
        <material name="table_mat" rgba=".8 .6 .4 1"/>
        <material name="ball_mat"  rgba="1 .2 .2 1"/>
        <material name="bowl_mat"  rgba=".9 .9 .9 1"/>{cam_marker_materials}
      </asset>

      <worldbody>
        <!-- Lighting -->
        <light name="top" pos="0 0 2" dir="0 0 -1" diffuse=".8 .8 .8"/>

        <!-- Overview camera (top-down) -->
        <camera name="overview_cam" pos="0 0 1.5" euler="0 0 0" fovy="60"/>{overview_marker_xml}

        <!-- Floor -->
        <geom name="floor" type="plane" size="2 2 .1" material="grid" condim="3"/>

        <!-- Table: top surface at z=0.5 -->
        <body name="table" pos="0 0 0.25">
          <geom name="table_top" type="box" size="0.4 0.3 0.025"
                pos="0 0 0.225" material="table_mat" condim="3"/>
          <geom name="leg_fl" type="cylinder" size="0.02 0.225" pos=" 0.35  0.25 0" material="table_mat"/>
          <geom name="leg_fr" type="cylinder" size="0.02 0.225" pos=" 0.35 -0.25 0" material="table_mat"/>
          <geom name="leg_bl" type="cylinder" size="0.02 0.225" pos="-0.35  0.25 0" material="table_mat"/>
          <geom name="leg_br" type="cylinder" size="0.02 0.225" pos="-0.35 -0.25 0" material="table_mat"/>
        </body>

        <!-- Ball: sphere resting on table surface (z = 0.5 + radius = 0.525) -->
        <body name="ball" pos="0.1 -0.1 0.525">
          <freejoint/>
          <geom name="ball_geom" type="sphere" size="0.025"
                material="ball_mat" mass="0.05" condim="4" solimp=".99 .99 .01" solref=".01 1"/>
        </body>

        <!-- Bowl: box-wall approximation, placed on table surface -->
        <body name="bowl" pos="-0.1 -0.1 0.5">
          <freejoint/>
          <geom name="bowl_bottom" type="cylinder" size="0.05 0.005"
                material="bowl_mat" mass="0.1"/>
          <geom name="bowl_wall_f" type="box" size="0.005 0.05 0.02" pos=" 0.05 0 0.02" material="bowl_mat" mass="0.02"/>
          <geom name="bowl_wall_b" type="box" size="0.005 0.05 0.02" pos="-0.05 0 0.02" material="bowl_mat" mass="0.02"/>
          <geom name="bowl_wall_l" type="box" size="0.05 0.005 0.02" pos="0  0.05 0.02" material="bowl_mat" mass="0.02"/>
          <geom name="bowl_wall_r" type="box" size="0.05 0.005 0.02" pos="0 -0.05 0.02" material="bowl_mat" mass="0.02"/>
        </body>
      </worldbody>

      <!-- Robot arm: include by filename only (temp file lives in same dir) -->
      <include file="{robot_xml_name}"/>

    </mujoco>
  """)


def init_robot_without_scene():
  xml_path = so_arm101_mj_description.MJCF_PATH  # robot_descriptions provides this
  model = mujoco.MjModel.from_xml_path(xml_path)
  data = mujoco.MjData(model)


def _inject_gripper_camera(robot_xml_path: str, out_dir: str) -> str:
  """Parse robot XML, inject a gripper camera, and write to a temp file. Returns temp path."""
  tree = ET.parse(robot_xml_path)
  root = tree.getroot()

  # Find <body name="moving_jaw_so101_v1"> (the jaw tip) anywhere in the tree
  jaw_body = root.find(".//{*}body[@name='moving_jaw_so101_v1']") or root.find(".//body[@name='moving_jaw_so101_v1']")
  if jaw_body is None:
    raise RuntimeError("Could not find <body name='moving_jaw_so101_v1'> in robot XML. Check body names.")

  cam = ET.SubElement(jaw_body, "camera")
  cam.set("name", "gripper_cam")
  cam.set("pos", "0 0 0.05")
  cam.set("euler", "180 0 0")
  cam.set("fovy", "60")

  if SHOW_CAMERA_MARKERS:
    marker = ET.SubElement(jaw_body, "site")
    marker.set("name", "gripper_cam_marker")
    marker.set("pos", "0 0 0.05")
    marker.set("size", "0.015")
    marker.set("type", "sphere")
    marker.set("rgba", "1 0.5 0 0.9")
    marker.set("material", "cam_marker_gripper")

  with tempfile.NamedTemporaryFile(
      suffix=".xml", mode="wb", delete=False, dir=out_dir
  ) as f:
    tree.write(f, encoding="utf-8", xml_declaration=True)
    tmp_robot_path = f.name

  return tmp_robot_path


def _load_scene_model() -> mujoco.MjModel:
  robot_xml_path = so_arm101_mj_description.MJCF_PATH
  robot_xml_dir = os.path.dirname(robot_xml_path)

  tmp_robot_path = _inject_gripper_camera(robot_xml_path, robot_xml_dir)
  tmp_robot_name = os.path.basename(tmp_robot_path)
  scene_xml = _build_scene_xml(tmp_robot_name)

  # Write scene temp file into the robot's directory so <include file="name.xml"/> resolves correctly.
  with tempfile.NamedTemporaryFile(
      suffix=".xml", mode="w", delete=False, dir=robot_xml_dir
  ) as f:
    f.write(scene_xml)
    tmp_scene_path = f.name

  try:
    model = mujoco.MjModel.from_xml_path(tmp_scene_path)
  finally:
    os.unlink(tmp_scene_path)
    os.unlink(tmp_robot_path)

  return model


def _place_robot_on_table(model: mujoco.MjModel) -> None:
  # Move the robot's base body onto the table surface (z=0.5).
  # <include> merges robot bodies directly into worldbody, so we offset them here.
  SCENE_BODY_NAMES = {"world", "table", "ball", "bowl"}
  TABLE_TOP_Z = 0.5
  ROBOT_X = 0.0
  ROBOT_Y = 0.25   # north edge of table (table spans ±0.3 in Y)
  for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    parent = model.body_parentid[i]
    # Root-level robot bodies: direct children of worldbody (parent==0) not in our scene
    if parent == 0 and (name or "") not in SCENE_BODY_NAMES:
      model.body_pos[i, 0] = ROBOT_X
      model.body_pos[i, 1] = ROBOT_Y
      model.body_pos[i, 2] += TABLE_TOP_Z


def _get_controlled_joint_indices(model: mujoco.MjModel) -> list[int]:
  # Find which qpos indices correspond to arm joints.
  # Hinge joints are in qpos; gripper may be 1-2 joints depending on model.
  joint_names = []
  joint_qposadr = []
  for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    joint_names.append(name)
    joint_qposadr.append(model.jnt_qposadr[j])

  print("Joints in model:")
  for n, adr in zip(joint_names, joint_qposadr):
    print(f"  {(n or '<unnamed>'):30s} qpos_adr={adr}")

  # --- You must decide which joints your policy predicts ---
  # For example, if your SO-ARM-101 has 6 joints + gripper, pick those here by name.
  # Replace with the actual names printed above.
  controlled_joint_names = [
      "1", "2", "3", "4", "5", "6"
      # add gripper joint(s) if present
  ]

  name_to_adr = {n: adr for n, adr in zip(joint_names, joint_qposadr)}
  qpos_indices = [name_to_adr[n] for n in controlled_joint_names if n in name_to_adr]
  assert len(qpos_indices) > 0, "Could not find controlled joints. Update controlled_joint_names."

  return qpos_indices


def _display_worker(queue):
  """Runs in a separate process; displays camera frames with cv2."""
  import cv2
  while True:
    item = queue.get()
    if item is None:  # sentinel: time to stop
      break
    for title, img_rgb in item:
      cv2.imshow(title, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
  cv2.destroyAllWindows()


def run_policy_and_actuate_robot(data, device, text_ids, j_std, j_mean, policy, renderer, q_prev_state):
  """Render gripper cam, run policy forward pass, write result to data.ctrl."""
  renderer.update_scene(data, camera="gripper_cam")
  img_hwc = renderer.render().copy()  # HWC uint8 RGB

  img_t = preprocess_image(img_hwc, image_size=128).unsqueeze(0).to(device)

  with torch.no_grad():
    pred_norm = policy(img_t, text_ids).squeeze(0).cpu().numpy()  # [J] normalized

  q_des = pred_norm * j_std + j_mean  # denormalize to actual joint units

  # Exponential smoothing to reduce jitter
  if q_prev_state[0] is None:
    q_prev_state[0] = data.ctrl[:len(q_des)].copy().astype(np.float32)
  q_cmd = (1 - POLICY_ALPHA) * q_prev_state[0] + POLICY_ALPHA * q_des
  q_prev_state[0] = q_cmd

  # Write to actuators (position-controlled joints)
  num_ctrl = min(len(q_cmd), len(data.ctrl))
  data.ctrl[:num_ctrl] = q_cmd[:num_ctrl]


def _run_viewer_loop(model: mujoco.MjModel, data: mujoco.MjData, run_policy: bool = False) -> None:
  # Load policy if requested
  policy = None
  text_ids = None
  j_mean = None
  j_std = None
  device = None
  policy_renderer = None
  q_prev_state = [None]  # mutable container for smoothing state across steps

  if run_policy:
    print(f"Loading policy from {POLICY_RUN_DIR}...")
    policy, tokenizer, j_mean, j_std, device = load_policy(POLICY_RUN_DIR)
    text_ids = tokenizer.encode(POLICY_COMMAND, max_len=16).unsqueeze(0).to(device)
    policy_renderer = mujoco.Renderer(model, height=128, width=128)
    print("Policy loaded. Running inference loop.")

  print("Launching simulation...")

  # Two renderers — one per camera (for display)
  gripper_renderer = mujoco.Renderer(model, height=240, width=320)
  overview_renderer = mujoco.Renderer(model, height=240, width=320)

  # Spawn display subprocess (has its own main thread; cv2.imshow works there)
  display_queue = mp.Queue(maxsize=2)
  display_proc = mp.Process(target=_display_worker, args=(display_queue,), daemon=True)
  display_proc.start()

  try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
      mujoco.mj_forward(model, data)

      while viewer.is_running():
        if run_policy and policy is not None:
          run_policy_and_actuate_robot(data, device, text_ids, j_std, j_mean, policy, policy_renderer, q_prev_state)

        # Step physics (5 steps × 0.002s = 0.01s simulated per iter = real-time)
        for _ in range(5):
          mujoco.mj_step(model, data)

        viewer.sync()

        # Render both cameras for the display windows
        gripper_renderer.update_scene(data, camera="gripper_cam")
        gripper_img = gripper_renderer.render().copy()

        overview_renderer.update_scene(data, camera="overview_cam")
        overview_img = overview_renderer.render().copy()

        try:
          display_queue.put_nowait([
              ("Gripper Camera", gripper_img),
              ("Overview Camera", overview_img),
          ])
        except mp.queues.Full:
          pass

        time.sleep(0.01)
  finally:
    display_queue.put(None)   # stop signal
    display_proc.join(timeout=2)
    del gripper_renderer
    del overview_renderer
    if policy_renderer is not None:
      del policy_renderer


def run_sim_on_scene(run_policy: bool = False):
  model = _load_scene_model()
  _place_robot_on_table(model)

  data = mujoco.MjData(model)
  _run_viewer_loop(model, data, run_policy=run_policy)


def main():
  parser = argparse.ArgumentParser(description="TylerVLA simulation")
  parser.add_argument(
      "--policy",
      action="store_true",
      help=f"Run trained policy from {POLICY_RUN_DIR} (default: teleop mode)",
  )
  args = parser.parse_args()

  run_sim_on_scene(run_policy=args.policy)


if __name__ == "__main__":
  main()
