"""
Action detection on the locked face using 5-point landmarks.

Detects:
  - Head movement (left/right/up/down) from bbox centre displacement
  - Smile from mouth-width / face-width ratio crossing a threshold
  - Blink from eye-to-nose distance dropping then recovering (5pt proxy)

Used by: detect.py, recognize.py (demo)
"""

from __future__ import annotations

import time

import numpy as np

from .config import (
    ACTION_BLINK_DROP_RATIO,
    ACTION_COOLDOWN_S,
    ACTION_MOVE_THRESHOLD_PX,
    ACTION_SMILE_RATIO,
)


class ActionDetector:
    """Stateful per-face action detector (call update() each frame)."""

    def __init__(self):
        self.prev_center = None
        self.prev_mouth_open = None
        self._eye_baseline: float | None = None
        self._blink_armed = False
        self._last_action_time: dict[str, float] = {}

        self.move_thr = ACTION_MOVE_THRESHOLD_PX
        self.smile_thr = ACTION_SMILE_RATIO
        self.blink_drop_ratio = ACTION_BLINK_DROP_RATIO
        self.cooldown_s = ACTION_COOLDOWN_S

    def _cooldown_ok(self, action: str, now: float) -> bool:
        last = self._last_action_time.get(action, 0.0)
        if now - last < self.cooldown_s:
            return False
        self._last_action_time[action] = now
        return True

    def update(self, kps: np.ndarray, bbox: np.ndarray) -> list[str]:
        """
        Process one frame for the locked face.

        Args:
            kps:  (5, 2) landmarks [L_eye, R_eye, nose, L_mouth, R_mouth]
            bbox: [x1, y1, x2, y2]

        Returns:
            List of action strings, e.g. ["moved left"], ["smile"], ["blink"]
        """
        actions: list[str] = []
        now = time.time()
        k = kps.astype(np.float32)

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        curr_center = (cx, cy)

        if self.prev_center is not None:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]

            dirs: list[str] = []
            if dx > self.move_thr:
                dirs.append("right")
            elif dx < -self.move_thr:
                dirs.append("left")

            if dy > self.move_thr:
                dirs.append("down")
            elif dy < -self.move_thr:
                dirs.append("up")

            if dirs:
                action = "moved " + "/".join(dirs)
                if self._cooldown_ok(action, now):
                    actions.append(action)

        self.prev_center = curr_center

        face_width = max(1.0, float(bbox[2] - bbox[0]))
        face_height = max(1.0, float(bbox[3] - bbox[1]))
        mouth_width = float(np.linalg.norm(k[4] - k[3]))
        ratio = mouth_width / face_width

        if self.prev_mouth_open is not None:
            if (
                self.prev_mouth_open < self.smile_thr
                and ratio >= self.smile_thr
                and self._cooldown_ok("smile", now)
            ):
                actions.append("smile")

        self.prev_mouth_open = ratio

        left_en = float(np.linalg.norm(k[0] - k[2]))
        right_en = float(np.linalg.norm(k[1] - k[2]))
        eye_open = (left_en + right_en) * 0.5 / face_height

        if self._eye_baseline is None:
            self._eye_baseline = eye_open
        else:
            if not self._blink_armed:
                self._eye_baseline = 0.92 * self._eye_baseline + 0.08 * eye_open

            drop = (self._eye_baseline - eye_open) / max(self._eye_baseline, 1e-6)

            if not self._blink_armed and drop >= self.blink_drop_ratio:
                self._blink_armed = True
            elif self._blink_armed and drop <= self.blink_drop_ratio * 0.35:
                if self._cooldown_ok("blink", now):
                    actions.append("blink")
                self._blink_armed = False
                self._eye_baseline = eye_open

        return actions
