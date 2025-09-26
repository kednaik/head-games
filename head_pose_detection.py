import math

import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
import pyautogui

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

class HeadPose(Enum):
    LOOKING_UP_ROLLING_RIGHT = "Looking Up and Rolling Right"
    LOOKING_UP_ROLLING_LEFT = "Looking Up and Rolling Left"
    LOOKING_DOWN_ROLLING_RIGHT = "Looking Down and Rolling Right"
    LOOKING_DOWN_ROLLING_LEFT = "Looking Down and Rolling Left"
    LOOKING_AT_SCREEN = "Looking at Screen"
    ROLLING_RIGHT = "Rolling Right"
    ROLLING_LEFT = "Rolling Left"
    LOOKING_UP = "Looking Up"
    LOOKING_DOWN = "Looking Down"
    LOOKING_LEFT = "Looking Left"
    LOOKING_RIGHT = "Looking Right"

class MouthStatus(Enum):
    OPEN = "Open"
    CLOSED = "Closed"

current_state = HeadPose.LOOKING_AT_SCREEN
previous_state = HeadPose.LOOKING_AT_SCREEN

current_mouth_status = MouthStatus.CLOSED
previous_mouth_status = MouthStatus.CLOSED

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)


def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]
    :return: Angles in degrees for each axis
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a later selfie-view display
        # This is done first so that all subsequent processing is done on the flipped image
        image = cv2.flip(image, 1)

        # Convert the color space from BGR to RGB and get Mediapipe results
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        
        image.flags.writeable = True
        # Convert the color space from RGB to BGR to display well with Opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape
        face_coordination_in_image = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_coordination_in_image.append([x, y])

                face_coordination_in_image = np.array(face_coordination_in_image,
                                                    dtype=np.float64)

                # The camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                    [0, focal_length, h / 2],
                                    [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Use solvePnP function to get rotation vector
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                result = rotation_matrix_to_angles(rotation_matrix)
                for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                    k, v = info
                    text = f'{k}: {int(v)}'
                    cv2.putText(image, text, (20, i*30 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                    
                pitch, yaw, roll = result
                if roll < -10 and pitch > 15:
                    current_state = HeadPose.LOOKING_UP_ROLLING_LEFT
                elif roll > 10 and pitch > 15:
                    current_state = HeadPose.LOOKING_UP_ROLLING_RIGHT
                elif roll < -10 and pitch < -15:
                    current_state = HeadPose.LOOKING_DOWN_ROLLING_LEFT
                elif roll > 10 and pitch < -15:
                    current_state = HeadPose.LOOKING_DOWN_ROLLING_RIGHT
                elif roll < -10:
                    current_state = HeadPose.ROLLING_LEFT
                elif roll > 10:
                    current_state = HeadPose.ROLLING_RIGHT
                elif pitch < -15:
                    current_state = HeadPose.LOOKING_DOWN
                elif pitch > 15:
                    current_state = HeadPose.LOOKING_UP
                else:
                    current_state = HeadPose.LOOKING_AT_SCREEN

                if current_state != previous_state:
                    pyautogui.keyUp('up')
                    pyautogui.keyUp('down')
                    pyautogui.keyUp('left')
                    pyautogui.keyUp('right')


                    if current_state == HeadPose.LOOKING_UP_ROLLING_RIGHT:
                        pyautogui.keyDown('up')
                        pyautogui.keyDown('right')
                    elif current_state == HeadPose.LOOKING_UP_ROLLING_LEFT:
                        pyautogui.keyDown('up')
                        pyautogui.keyDown('left')
                    elif current_state == HeadPose.LOOKING_DOWN_ROLLING_RIGHT:
                        pyautogui.keyDown('down')
                        pyautogui.keyDown('right')
                    elif current_state == HeadPose.LOOKING_DOWN_ROLLING_LEFT:
                        pyautogui.keyDown('down')
                        pyautogui.keyDown('left')
                    elif current_state == HeadPose.LOOKING_UP:
                        pyautogui.keyDown('up')
                    elif current_state == HeadPose.LOOKING_DOWN:
                        pyautogui.keyDown('down')
                    elif current_state == HeadPose.ROLLING_LEFT:
                        pyautogui.keyDown('left')
                    elif current_state == HeadPose.ROLLING_RIGHT:
                        pyautogui.keyDown('right')

                previous_state = current_state
                cv2.putText(image, f'Head Pose: {current_state.value}', (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                lip_upper = face_landmarks.landmark[13]
                lip_lower = face_landmarks.landmark[14]
                lip_distance = math.hypot(lip_upper.x - lip_lower.x, lip_upper.y - lip_lower.y)
                
                if lip_distance > 0.04:
                    current_mouth_status = MouthStatus.OPEN
                else:
                    current_mouth_status = MouthStatus.CLOSED

                if current_mouth_status != previous_mouth_status:
                    pyautogui.keyUp('h')
                    previous_mouth_status = current_mouth_status
                    if current_mouth_status == MouthStatus.OPEN:
                        pyautogui.keyDown('h')

                cv2.putText(image, f'Mouth: {current_mouth_status}', (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        cv2.imshow('Head Pose Angles', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
