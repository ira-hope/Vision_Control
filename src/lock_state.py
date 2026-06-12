"""Live lock status written by detect.py and read by the dashboard."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import LOCK_STATE_PATH


def write_lock_state(
    *,
    target_name: str,
    locked_name: str | None = None,
    is_locked: bool = False,
    last_action: str = "",
    confidence: float | None = None,
    motor_command: str = "",
) -> None:
    """Persist current lock status for the HTML dashboard."""
    LOCK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "target_name": target_name,
        "locked_name": locked_name,
        "is_locked": is_locked,
        "last_action": last_action,
        "motor_command": motor_command,
        "updated_at": time.time(),
        "updated_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if confidence is not None:
        state["confidence"] = round(float(confidence), 4)
    LOCK_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def read_lock_state() -> dict[str, Any]:
    """Return lock status; defaults when detect.py is not running."""
    if not LOCK_STATE_PATH.exists():
        return {
            "target_name": "",
            "locked_name": None,
            "is_locked": False,
            "last_action": "",
            "motor_command": "",
            "confidence": None,
            "updated_at": 0,
            "updated_iso": "",
        }
    try:
        return json.loads(LOCK_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "target_name": "",
            "locked_name": None,
            "is_locked": False,
            "last_action": "",
            "motor_command": "",
            "confidence": None,
            "updated_at": 0,
            "updated_iso": "",
        }
