import numpy as np

def detect_horizontal_movement(prev_center, curr_center, threshold=15):
    dx = curr_center[0] - prev_center[0]
    if dx > threshold:
        return "MOVE_RIGHT"
    if dx < -threshold:
        return "MOVE_LEFT"
    return None


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def detect_blink(ear, blink_threshold=0.20):
    return ear < blink_threshold


def detect_smile(mouth_width, face_width, ratio=0.45):
    return (mouth_width / face_width) > ratio
