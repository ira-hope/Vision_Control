from .face_tracker import FaceTracker
from .history_logger import start_history, log_action
from .actions import detect_horizontal_movement

tracker = FaceTracker(target_name="Nkerabahizi")
history_file = None
prev_center = None


def handle_face(name, bbox):
    global history_file, prev_center

    status = tracker.update(name, bbox)

    if status == "LOCKED" and history_file is None:
        history_file = start_history(name)
        log_action(history_file, "LOCKED", "Face successfully locked")

    if tracker.locked:
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        current_center = (cx, cy)

        if prev_center:
            movement = detect_horizontal_movement(prev_center, current_center)
            if movement:
                log_action(history_file, movement)

        prev_center = current_center
