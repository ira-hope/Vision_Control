import json
import time
from pathlib import Path
from typing import Optional

class HistoryManager:
    def __init__(self, log_dir: str = "data/history"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "history_log.jsonl"
        self.buffer = []

    def log_event(self, name: str, action: str):
        """Log a new event with current timestamp."""
        event = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "action": action
        }
        self.buffer.append(event)
        
        # Immediate write for safety, can be buffered if performance needed
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def get_recent_events(self, limit: int = 10):
        return self.buffer[-limit:]
