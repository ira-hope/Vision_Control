"""

Shared project configuration: paths, thresholds, and constants.



BENAX assessment tunables (MQTT broker, recognition threshold, deadband, etc.)

are centralised here. Edit this file rather than hardcoding in individual scripts.



Run from project root with .venv active (Python 3.10+; tested on 3.12).

"""



from __future__ import annotations



import sys

from pathlib import Path

from typing import Sequence, Tuple



import cv2



PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEDIAPIPE_VERSION = "0.10.21"



# --- Single-identity recognition (speaker lock) ---

DISTANCE_THRESHOLD = 0.48

MIN_RECOGNITION_SIMILARITY = 0.50

MATCH_MARGIN = 0.06

RECOGNITION_VOTE_WINDOW = 7

ALIGN_SIZE = (112, 112)



# --- Face tracking & motor commands (assessment c) ---

TRACKING_THRESHOLD_PX = 50.0

TRACKING_CENTER_DEADBAND_RATIO = 0.06  # ~10 px on 160px-wide frame

TRACKING_CENTER_HYSTERESIS_RATIO = 0.025

TRACKING_CENTER_DEADBAND_MIN_PX = 6.0
TRACKING_MOVE_MIN_PX = 2.5  # min horizontal shift per frame before panning (~160px cam)

MOTOR_PAN_INVERT = False  # set True if left/right are reversed on the servo mount

# Servo angle tracking — face offset from centre -> pan angle
SERVO_ANGLE_MIN = 0
SERVO_ANGLE_MAX = 180
SERVO_CENTER_ANGLE = 90
SERVO_MAX_PAN_DEG = 32  # max pan from centre when face is at frame edge
SERVO_TRACKING_RANGE_RATIO = 0.9
SERVO_FACE_SMOOTH_ALPHA = 0.22  # lower = less jitter from landmark noise
SERVO_SMOOTHING_FACTOR = 0.15  # lower = slower, smoother follow
SERVO_MAX_STEP_DEG = 3  # degrees per MQTT update while following
SERVO_CATCHUP_LAG_DEG = 8  # if lag exceeds this, use faster catch-up steps
SERVO_CATCHUP_STEP_DEG = 8  # catch-up step after SCAN / large lag
SERVO_INVERTED = MOTOR_PAN_INVERT
SERVO_DEADZONE_DEG = 2
SERVO_PUBLISH_INTERVAL_MS = 100
SERVO_FACE_SMOOTH_ALPHA = 0.35
SERVO_DEBUG = True



# --- Enrollment (assessment a: 10–30 facial images) ---

ENROLL_SAMPLES_MIN = 10

ENROLL_SAMPLES_MAX = 30

ENROLL_SAMPLES_DEFAULT = 15



# --- Action detection ---

ACTION_MOVE_THRESHOLD_PX = 15.0

ACTION_SMILE_RATIO = 0.40

ACTION_BLINK_DROP_RATIO = 0.12

ACTION_COOLDOWN_S = 0.9



# --- Camera ---

CAMERA_INDEX = 1

CAMERA_INDICES: Tuple[int, ...] = (CAMERA_INDEX,)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# 0 = show at capture size (1:1, no upscale)
PREVIEW_WIDTH = 0
PREVIEW_HEIGHT = 0
PREVIEW_WINDOW = "Face Locking with Servo Control"




# --- MQTT / ESP8266 (assessment d) ---

MQTT_BROKER = "157.173.101.159"

MQTT_PORT = 1883

MQTT_TOPIC_MOTOR_CMD = "TeAmSiX/facelocking/servo_ctrl_x9z"

MQTT_TOPIC_SERVO_ANGLE = MQTT_TOPIC_MOTOR_CMD

ESP8266_SEARCH_TIMEOUT_MS = 1500  # must match servo_controller.ino

LOCK_RELEASE_FRAMES = 90



# --- MediaPipe ---

MIN_DETECTION_CONFIDENCE = 0.5

MIN_TRACKING_CONFIDENCE = 0.5





def project_path(*parts: str) -> Path:

    return PROJECT_ROOT.joinpath(*parts)





DB_PATH = project_path("data", "db", "face_db.npz")

DB_JSON_PATH = project_path("data", "db", "face_db.json")

ENROLL_DIR = project_path("data", "enroll")

HISTORY_DIR = project_path("data", "history")

HISTORY_LOG_PATH = project_path("data", "history", "history_log.jsonl")

LOCK_STATE_PATH = project_path("data", "history", "lock_state.json")

DASHBOARD_DIR = project_path("dashboard")

MODEL_PATH = project_path("models", "embedder_arcface.onnx")





def open_camera(indices: Sequence[int] = CAMERA_INDICES) -> cv2.VideoCapture:

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0

    for idx in indices:

        cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAMERA_WIDTH))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAMERA_HEIGHT))
            ret, _ = cap.read()
            if ret:
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(
                    f"Camera opened on index {idx} ({aw}x{ah}, "
                    f"preview native 1:1)."
                )
                return cap

        cap.release()

    raise RuntimeError(f"Camera not opened. Tried indices {list(indices)}.")





def scale_preview(frame):
    """Optional upscale; returns frame unchanged when PREVIEW size is 0."""
    if PREVIEW_WIDTH <= 0 or PREVIEW_HEIGHT <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == PREVIEW_WIDTH and h == PREVIEW_HEIGHT:
        return frame
    return cv2.resize(
        frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_LINEAR
    )
