import os
from datetime import datetime

def start_history(face_name):
    os.makedirs("history", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{face_name}_history_{timestamp}.txt"
    return open(os.path.join("history", filename), "a")


def log_action(file, action, details=""):
    time = datetime.now().strftime("%H:%M:%S")
    file.write(f"[{time}] {action} {details}\n")
    file.flush()
