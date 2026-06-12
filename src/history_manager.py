"""
JSONL event logger for face locking actions.

Writes one JSON object per line to data/history/history_log.jsonl.
Used by detect.py to record LOCKED, moved left/right, smile, etc.

Each event: { timestamp, iso_time, name, action }
"""

import json
import time
from pathlib import Path

from .config import HISTORY_DIR, HISTORY_LOG_PATH


class HistoryManager:
    """Append-only logger for lock and action events."""

    def __init__(self, log_dir: Path = HISTORY_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "history_log.jsonl"
        self.buffer = []  # in-memory copy for get_recent_events()

    def log_event(self, name: str, action: str):
        """Log a new event with current timestamp and flush to disk immediately."""
        event = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "action": action,
        }
        self.buffer.append(event)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def get_recent_events(self, limit: int = 10):
        """Return the last N events from the in-memory buffer."""
        return self.buffer[-limit:]


def read_events_from_disk(limit: int | None = None) -> list[dict]:
    """Load events from history_log.jsonl (newest last)."""
    if not HISTORY_LOG_PATH.exists():
        return []

    events: list[dict] = []
    try:
        with open(HISTORY_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    if limit is not None and limit > 0:
        return events[-limit:]
    return events
