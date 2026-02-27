import time

class FaceTracker:
    def __init__(self, target_name, max_missing_time=2.0):
        self.target_name = target_name
        self.locked = False
        self.last_seen_time = None
        self.last_bbox = None
        self.max_missing_time = max_missing_time

    def update(self, recognized_name, bbox):
        now = time.time()

        if recognized_name == self.target_name:
            self.locked = True
            self.last_seen_time = now
            self.last_bbox = bbox
            return "LOCKED"

        if self.locked:
            if now - self.last_seen_time <= self.max_missing_time:
                return "TRACKING"
            else:
                self.reset()
                return "UNLOCKED"

        return "SEARCHING"

    def reset(self):
        self.locked = False
        self.last_seen_time = None
        self.last_bbox = None
