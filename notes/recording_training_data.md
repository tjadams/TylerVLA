# recording_training_data
7-14-26:
- An iPhone on a tripod that isn't connected to a Mac, can't be used as an observation camera for a LeRobot dataset. LeRobot grabs the frames from OpenCV dand timestamps them against motor reads. It needs the camera connected ot the computer, not an independently recorded video file.
- Solution for simulated training data: TBD
- Solution for real training data: Use the Continuity Camera feature to make the iPhone a wireless webcam for the Mac.

Steps:
1. Enable Continuity Camera on iPhone: Settings → General → AirPlay & Continuity
2. Set up packages: `conda activate lerobot`
3. Confirm camera is set up: `python real_robot/recording_training_data/probe_cameras.py`
4. Confirm 2 robots are connected: `ls /dev/tty.usbmodem*` (also finds 2 ports for next script)
5. Record: `real_robot/recording_training_data/record_episode.sh`

Controls during recording: 
1. Right arrow: end episode
2. Left arrow: re-record current episode
3. Escape: finish recording (encodes video, writes metadata)
4. Ctrl+C: breaks recording and leaves partial files

Bug: re-record episode with left arrow results in this error.
`OSError: [Errno 66] Directory not empty: '/Users/tjadams/tylervla_datasets/pick-place_20260718_124803/images/observation.images.front/episode_000001'`