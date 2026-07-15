# recording_training_data
7-14-26:
- An iPhone on a tripod that isn't connected to a Mac, can't be used as an observation camera for a LeRobot dataset. LeRobot grabs the frames from OpenCV dand timestamps them against motor reads. It needs the camera connected ot the computer, not an independently recorded video file.
- Solution for simulated training data: TBD
- Solution for real training data: Use the Continuity Camera feature to make the iPhone a wireless webcam for the Mac.

Steps:
1. Enable Continuity Camera on iPhone: Settings → General → AirPlay & Continuity
2. conda activate lerobot
3. (Optional) probe_cameras.py
4. (Optional) ls /dev/tty.usbmodem* to find the FOLLOWER_PORT/LEADER_PORT for putting in the next script
5. real_robot/recording_training_data/record_episode.sh

Controls while recording (keys go to the --display_data/rerun window, not the terminal):
- → : end the current episode early and advance (normal "done with this demo")
- ← : discard and re-record the current episode (fumbled the grasp)
- Esc : stop the whole session cleanly — encodes videos, writes metadata (the correct way to quit)

Records NUM_EPISODES back-to-back; each runs ≤ episode_time_s (default 60s) with a reset window between to reposition the ball. Never Ctrl+C — it skips encoding and leaves partial files.
macOS: these keys use pynput, which needs Accessibility permission. Grant your terminal under System Settings → Privacy & Security → Accessibility.
