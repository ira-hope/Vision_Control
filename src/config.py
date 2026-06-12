"""
Shared project configuration: paths, thresholds, and constants.

All modules import paths and tunables from here so they work regardless of
the current working directory. Edit DISTANCE_THRESHOLD, MQTT_BROKER, etc.
in this file rather than hardcoding values in individual scripts.

Run from project root with .venv active (Python 3.12).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2

# Project root (Facelocking2/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Recommended MediaPipe version (see README)
MEDIAPIPE_VERSION = "0.10.21"

# Recognition — tuned so enrolled identities rarely flicker to Unknown
DISTANCE_THRESHOLD = 0.48
MIN_RECOGNITION_SIMILARITY = 0.50
MATCH_MARGIN = 0.06
RECOGNITION_VOTE_WINDOW = 7

ALIGN_SIZE = (112, 112)

# Face locking
LOCK_GRACE_S = 2.0
TRACKING_THRESHOLD_PX = 50.0

# Action detection (ActionDetector)
ACTION_MOVE_THRESHOLD_PX = 15.0
ACTION_SMILE_RATIO = 0.40
ACTION_BLINK_DROP_RATIO = 0.12
ACTION_COOLDOWN_S = 0.9

# Camera
CAMERA_INDICES: Tuple[int, ...] = (0, 1, 2)

# MQTT / servo (faceLockServo.py)
MQTT_BROKER = "157.173.101.159"
MQTT_PORT = 1883
MQTT_TOPIC_SERVO_ANGLE = "TeAmSiX/facelocking/servo_ctrl_x9z"
SERVO_ANGLE_MIN = 0
SERVO_ANGLE_MAX = 180
SERVO_SMOOTHING_FACTOR = 0.3
SERVO_INVERTED = False
SERVO_DEADZONE_DEG = 5
SERVO_PUBLISH_INTERVAL_MS = 100
LOCK_RELEASE_FRAMES = 30

# MediaPipe detection
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


def project_path(*parts: str) -> Path:
    """Build an absolute path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)


# Data paths
DB_PATH = project_path("data", "db", "face_db.npz")
DB_JSON_PATH = project_path("data", "db", "face_db.json")
ENROLL_DIR = project_path("data", "enroll")
HISTORY_DIR = project_path("data", "history")
HISTORY_LOG_PATH = project_path("data", "history", "history_log.jsonl")
LOCK_STATE_PATH = project_path("data", "history", "lock_state.json")
DASHBOARD_DIR = project_path("dashboard")
MODEL_PATH = project_path("models", "embedder_arcface.onnx")


def open_camera(indices: Sequence[int] = CAMERA_INDICES) -> cv2.VideoCapture:
    """Try camera indices in order; return the first opened capture."""
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened on index {idx}.")
            return cap
        cap.release()
    raise RuntimeError(f"Camera not opened. Tried indices {list(indices)}.")
