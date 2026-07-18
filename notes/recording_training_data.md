# recording_training_data
## Overview
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

## 7-18-26: session notes
Results: 5 successful episodes, bugfixes, tuning scripts, human-robot-control skill improvements, compression script

Improvements this session:
- My skills in controlling the leader to implicitly control the follower, improved
- Putting laptop down on my workout bench is a good place for it
- Created a compression script using ffmpeg, reduced from 150mb to 9mb mp4s

Improvements for future sessions:
- Not yet, but perhaps longer usb-c cables
- Bluetooth keyboard because when I want to press keyboard controls during recording, I move in front of the camera
- Call compression script (real_robot/utils/compress_video.sh) in record_episode.sh

Issue observed: I controlled the follower perhaps a bit too hard on the leader size when squeezing the medicine container, and it got upset and went fully open on the end effector. Then I saw this log: RuntimeError: Failed to write 'Torque_Enable' on id_=6 with '0' after 6 tries. [RxPacketError] Overload error!

Bug fixed: re-record episode with left arrow results in this error.
`OSError: [Errno 66] Directory not empty: '/Users/tjadams/tylervla_datasets/pick-place_20260718_124803/images/observation.images.front/episode_000001'`

## 7-14-26: session notes
- An iPhone on a tripod that isn't connected to a Mac, can't be used as an observation camera for a LeRobot dataset. LeRobot grabs the frames from OpenCV dand timestamps them against motor reads. It needs the camera connected ot the computer, not an independently recorded video file.
- Solution for simulated training data: TBD
- Solution for real training data: Use the Continuity Camera feature to make the iPhone a wireless webcam for the Mac.