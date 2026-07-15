# recording_training_data
7-14-26:
- An iPhone on a tripod that isn't connected to a Mac, can't be used as an observation camera for a LeRobot dataset. LeRobot grabs the frames from OpenCV dand timestamps them against motor reads. It needs the camera connected ot the computer, not an independently recorded video file.
- Solution for simulated training data: TBD
- Solution for real training data: Use the Continuity Camera feature to make the iPhone a wireless webcam for the Mac.

Steps:
1. Enable Continuity Camera on iPhone: Settings → General → AirPlay & Continuity
1. conda activate lerobot
2. (Optional) probe_cameras.py
2. (Optional) ls /dev/tty.usbmodem* to find the FOLLOWER_PORT/LEADER_PORT for putting in the next script
2. real_robot/recording_training_data/record_episode.sh