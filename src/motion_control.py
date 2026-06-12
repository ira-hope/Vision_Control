"""
Map locked-face horizontal position to servo angle (0-180 degrees).

Uses offset from frame centre so head movement maps to pan angle.
On lock / re-acquire after SCAN, snaps immediately to the face angle.
While tracking, steps toward the target each publish interval.
"""

from __future__ import annotations

import time
from typing import Optional

from .config import (
    SERVO_ANGLE_MAX,
    SERVO_ANGLE_MIN,
    SERVO_CATCHUP_LAG_DEG,
    SERVO_CATCHUP_STEP_DEG,
    SERVO_CENTER_ANGLE,
    SERVO_DEADZONE_DEG,
    SERVO_FACE_SMOOTH_ALPHA,
    SERVO_INVERTED,
    SERVO_MAX_PAN_DEG,
    SERVO_MAX_STEP_DEG,
    SERVO_PUBLISH_INTERVAL_MS,
    SERVO_SMOOTHING_FACTOR,
    SERVO_TRACKING_RANGE_RATIO,
)

CMD_SCAN = "SCAN"
CMD_OUT_OF_FRAME = "OUT_OF_FRAME"


def is_scan_command(command: str) -> bool:
    return command in {CMD_SCAN, CMD_OUT_OF_FRAME}


def format_motor_display(command: str) -> str:
    if command and command.isdigit():
        return f"{command}°"
    return command


def format_angle_value(angle: float) -> str:
    return f"{int(round(angle))}°"


def _clamp_angle(angle: float) -> float:
    return max(float(SERVO_ANGLE_MIN), min(float(SERVO_ANGLE_MAX), angle))


def _step_toward(current: float, target: float, max_step: float) -> float:
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + max_step if delta > 0 else current - max_step


class ServoAngleTracker:
    """Convert smoothed face X position to a gradual servo angle."""

    def __init__(self, frame_width: int):
        self.frame_width = max(1, frame_width)
        self.frame_center_x = self.frame_width / 2.0
        self.smoothed_face_x: Optional[float] = None
        self.normalized_offset = 0.0
        self.current_angle = float(SERVO_CENTER_ANGLE)
        self.target_angle = float(SERVO_CENTER_ANGLE)
        self.last_published_angle = float(SERVO_CENTER_ANGLE)
        self.last_publish_time = 0.0
        self.last_command = CMD_SCAN

    def reset_anchor(self, face_center_x: Optional[float] = None) -> None:
        if face_center_x is None:
            return
        self.smoothed_face_x = float(face_center_x)
        self.target_angle = self._face_x_to_angle(self.smoothed_face_x)
        self.current_angle = self.target_angle

    def snap_to_face(self, face_center_x: float) -> str:
        """Align servo command to the face immediately (after SCAN / re-acquire)."""
        self.smoothed_face_x = float(face_center_x)
        self.target_angle = self._face_x_to_angle(self.smoothed_face_x)
        self.current_angle = self.target_angle
        angle_int = int(round(self.target_angle))
        self.last_published_angle = float(angle_int)
        self.last_publish_time = time.time() * 1000
        self.last_command = str(angle_int)
        return self.last_command

    def _face_x_to_angle(self, face_x: float) -> float:
        half_span = max(1.0, (self.frame_width / 2.0) * SERVO_TRACKING_RANGE_RATIO)
        self.normalized_offset = (face_x - self.frame_center_x) / half_span
        self.normalized_offset = max(-1.0, min(1.0, self.normalized_offset))

        pan = -self.normalized_offset * float(SERVO_MAX_PAN_DEG)
        if SERVO_INVERTED:
            pan = -pan
        return _clamp_angle(SERVO_CENTER_ANGLE + pan)

    def _smooth_face_x(self, face_center_x: float) -> float:
        if self.smoothed_face_x is None:
            self.smoothed_face_x = face_center_x
        else:
            self.smoothed_face_x += (
                face_center_x - self.smoothed_face_x
            ) * SERVO_FACE_SMOOTH_ALPHA
        return self.smoothed_face_x

    def _next_publish_angle(self) -> int:
        lag = abs(self.target_angle - self.last_published_angle)
        max_step = (
            SERVO_CATCHUP_STEP_DEG
            if lag >= float(SERVO_CATCHUP_LAG_DEG)
            else SERVO_MAX_STEP_DEG
        )
        stepped = _step_toward(
            self.last_published_angle,
            self.target_angle,
            float(max_step),
        )
        return int(round(_clamp_angle(stepped)))

    def debug_line(self) -> str:
        face_x = f"{self.smoothed_face_x:.0f}" if self.smoothed_face_x is not None else "—"
        lag = self.target_angle - self.last_published_angle
        return (
            f"face_x={face_x} offset={self.normalized_offset:+.2f} "
            f"target={format_angle_value(self.target_angle)} "
            f"servo={format_angle_value(self.last_published_angle)} "
            f"lag={lag:+.0f}°"
        )

    def update(
        self,
        face_center_x: Optional[float],
        *,
        searching: bool = False,
        force: bool = False,
    ) -> Optional[str]:
        if searching or face_center_x is None:
            self.last_command = CMD_SCAN
            return CMD_SCAN

        if force:
            return self.snap_to_face(face_center_x)

        smooth_x = self._smooth_face_x(face_center_x)
        self.target_angle = self._face_x_to_angle(smooth_x)
        self.current_angle += (
            self.target_angle - self.current_angle
        ) * SERVO_SMOOTHING_FACTOR

        publish_angle = self._next_publish_angle()
        now_ms = time.time() * 1000

        lag = abs(self.target_angle - self.last_published_angle)
        moved = lag >= float(SERVO_DEADZONE_DEG)
        changed = publish_angle != int(round(self.last_published_angle))
        rate_ok = now_ms - self.last_publish_time >= SERVO_PUBLISH_INTERVAL_MS

        if not moved or not changed or not rate_ok:
            self.last_command = str(int(round(self.last_published_angle)))
            return None

        self.last_published_angle = float(publish_angle)
        self.last_publish_time = now_ms
        self.last_command = str(publish_angle)
        return self.last_command
