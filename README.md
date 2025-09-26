# Head Games: A Python-Powered Face Gesture Control System for gaming*

Welcome to **Face-Controlled Racing**, a Python project that lets you play *Mr. Racer - Car Racing* on [Poki](https://poki.com/en/g/mr-racer-car-racing) using face gestures! By leveraging MediaPipe’s FaceMesh for facial landmark detection, OpenCV for head pose estimation, and PyAutoGUI for keyboard input simulation, this project transforms your webcam into a hands-free game controller. Tilt your head to steer, look up or down to control speed, and open your mouth to honk the horn—perfect for an immersive, accessible racing experience.

## Features
- **Head Pose Controls**:
  - **Look Up (pitch > 15°)**: Accelerates (up arrow), speeding through *Mr. Racer*’s highways.
  - **Look Down (pitch < -15°)**: Brakes or reverses (down arrow), ideal for avoiding collisions.
  - **Tilt Left (roll < -10°)**: Steers left (left arrow), dodging traffic.
  - **Tilt Right (roll > 10°)**: Steers right (right arrow), navigating curves.
  - **Combined Poses**: E.g., looking up and tilting left accelerates while steering left for smooth overtakes.
- **Mouth Detection**: Open your mouth (lip distance > 0.04) to press ‘H’ and blare the horn, adding flair to races.
- **Real-Time Feedback**: Displays head pose angles (pitch, yaw, roll), current state (e.g., “Looking Up and Rolling Right”), and mouth status (“Open”/“Closed”) on an OpenCV window with a facial mesh overlay.
- **Accessibility**: Enables hands-free gaming, aligning with trends like Google’s Project Gameface for inclusive interfaces.

## Demo
![Head Games in Action](Game_Screen_Recording_GIF.gif)

## Prerequisites
- **Hardware**: A computer with a webcam (built-in or external).
- **Software**:
  - Python 3.8 or higher.
  - Libraries: `opencv-python`, `mediapipe`, `numpy`, `pyautogui`.
  - A modern web browser (Google Chrome recommended for Poki).
- **Environment**: Good, even lighting (avoid backlighting or shadows) for accurate face detection.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kednaik/head-games.git
   cd face-controlled-racing
   ```
2. **Install Dependencies**:
   Install required Python libraries using pip:
   ```bash
   pip install opencv-python mediapipe numpy pyautogui
   ```
3. **Verify Webcam**: Ensure your webcam is functional (test in a video app like Zoom or Windows Camera).
4. **Download Game**: No download needed! *Mr. Racer* runs in your browser at [https://poki.com/en/g/mr-racer-car-racing](https://poki.com/en/g/mr-racer-car-racing).

## Usage
1. **Position Yourself**: Sit 1–2 feet from the webcam, facing it directly, with even lighting (e.g., a desk lamp in front).
2. **Run the Script**:
   - Save the main script as `face_racer.py`.
   - Open a terminal in the project directory and run:
     ```bash
     python face_racer.py
     ```
   - An OpenCV window (“Head Pose Angles”) will open, showing your webcam feed with facial landmarks and pose/mouth status.
3. **Open *Mr. Racer***:
   - In Google Chrome, navigate to [https://poki.com/en/g/mr-racer-car-racing](https://poki.com/en/g/mr-racer-car-racing).
   - Click “Play” and select a mode (Free Roam is great for practice).
   - Press `F11` for fullscreen to enhance immersion.
4. **Focus the Game**: Click the browser window to ensure PyAutoGUI sends inputs to *Mr. Racer*.
5. **Race with Your Face**:
   - **Tilt head left/right**: Steer (left/right arrows).
   - **Look up/down**: Accelerate (up arrow) or brake (down arrow).
   - **Open mouth**: Honk the horn (‘H’ key).
   - Example: Tilt left and look up to speed around a car while turning.
6. **Monitor Feedback**: Watch the OpenCV window for real-time pose data (e.g., “Head Pose: Looking Up and Rolling Right”).
7. **Quit**: Press `Esc` in the OpenCV window to stop the script.

## How It Works
The project uses a pipeline to translate face gestures into game controls:
- **Webcam Capture**: OpenCV grabs video frames, flipped for selfie-view and converted to RGB.
- **Facial Landmarks**: MediaPipe’s FaceMesh detects 468 landmarks, with six key points (indices 1, 9, 57, 130, 287, 359) used for pose estimation.
- **Head Pose Estimation**: OpenCV’s `solvePnP` maps 2D landmarks to a 3D face model, yielding pitch (up/down) and roll (left/right) angles via a rotation matrix.
  ```python
  success, rotation_vec, transition_vec = cv2.solvePnP(
      face_coordination_in_real_world, face_coordination_in_image, cam_matrix, dist_matrix)
  rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)
  pitch, yaw, roll = rotation_matrix_to_angles(rotation_matrix)
  ```
- **Control Mapping**: Head poses trigger arrow keys (e.g., pitch > 15° for up arrow, roll < -10° for left arrow), with combined poses for diagonal moves.
  ```python
  if roll < -10 and pitch > 15:
      current_state = HeadPose.LOOKING_UP_ROLLING_LEFT
      pyautogui.keyDown('up')
      pyautogui.keyDown('left')
  ```
- **Mouth Detection**: Lip distance > 0.04 presses ‘H’ for the horn.
  ```python
  lip_distance = math.hypot(lip_upper.x - lip_lower.x, lip_upper.y - lip_lower.y)
  if lip_distance > 0.04:
      current_mouth_status = MouthStatus.OPEN
      pyautogui.keyDown('h')
  ```
- **Visualization**: Overlays pose angles, state, and mouth status on the webcam feed.
  ```python
  cv2.putText(image, f'Head Pose: {current_state.value}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
  ```

## Troubleshooting
- **No Face Detected**: Adjust lighting (avoid bright lights behind you) or move closer to the webcam.
- **Inputs Not Registering**: Ensure the *Mr. Racer* browser window is focused (click it after starting the script).
- **Laggy Performance**: Close other apps to free CPU resources; ensure your webcam supports 30 FPS.
- **Accidental Inputs**: PyAutoGUI’s failsafe (move mouse to top-left corner) stops runaway inputs.

## Future Enhancements
- Add a calibration step to personalize pitch/roll thresholds.
- Smooth angles with a moving average for jitter-free steering.
- Support mouse control for camera panning in other games.
- Extend to other Poki games with similar controls (e.g., *Rally Point*).

## Contributing
Contributions are welcome! Fork the repo, make improvements (e.g., smoother controls, new game support), and submit a pull request. Ideas:
- Add dynamic threshold calibration.
- Integrate with other racing games.
- Optimize for lower-end hardware.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- Built with [MediaPipe](https://mediapipe.dev/) for face tracking, [OpenCV](https://opencv.org/) for pose estimation, and [PyAutoGUI](https://pyautogui.readthedocs.io/) for input simulation.
- Inspired by the growing gesture recognition market and projects like Google’s Project Gameface for accessible gaming.
- Thanks to Poki for hosting *Mr. Racer*, a perfect testbed for face-controlled racing!

Happy racing! 🚗 Share your high scores or fork the project to create your own face-controlled gaming adventure.