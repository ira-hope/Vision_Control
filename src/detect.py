"""
Main face locking application.

Pipeline per frame:
  1. Detect all faces (HaarFaceMesh5pt)
  2. Align + embed + match each face against face_db.npz
  3. Lock onto a chosen target (--name or prompt)
  4. Track with identity match + spatial fallback if ID flickers
  5. Run ActionDetector on locked face; log events to history_log.jsonl

Run:
  python -m src.detect
  python -m src.detect --name Alice

Quit: q
Colors: yellow = locked target, green = known other, red = unknown
"""

import argparse
import time

import cv2
import numpy as np

from .action_detector import ActionDetector
from .config import (
    ALIGN_SIZE,
    DB_PATH,
    LOCK_GRACE_S,
    TRACKING_THRESHOLD_PX,
    open_camera,
)
from .embed import ArcFaceEmbedderONNX
from .haar_5pt import HaarFaceMesh5pt, align_face_5pt
from .history_manager import HistoryManager
from .lock_state import write_lock_state
from .recognize import FaceDBMatcher, RecognitionStabilizer, load_db_npz


def _resolve_target_name(names: list[str], cli_name: str | None) -> str:
    """Pick lock target from --name flag or interactive prompt."""
    if not names:
        raise RuntimeError("No enrolled identities. Run: python -m src.enroll")

    if cli_name:
        if cli_name not in names:
            raise RuntimeError(f"'{cli_name}' not in database. Choose from: {names}")
        return cli_name

    print("\nEnrolled identities:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")

    print("\nEnter the name of the identity to lock (exact match): ", end="")
    try:
        choice = input().strip()
    except EOFError:
        choice = names[0]

    if not choice:
        choice = names[0]
    if choice not in names:
        raise RuntimeError(f"'{choice}' not in database. Choose from: {names}")
    return choice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face recognition + face locking")
    parser.add_argument(
        "--name", "-n",
        help="Target identity to lock onto (must be enrolled in the DB)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db = load_db_npz(DB_PATH)
    target_name = _resolve_target_name(sorted(db.keys()), args.name)
    print(f"Will lock onto: {target_name}")
    write_lock_state(target_name=target_name, is_locked=False)

    # Core pipeline components
    det = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher.from_disk(DB_PATH, embedder=embedder)
    stabilizer = RecognitionStabilizer(set(db.keys()))
    history_manager = HistoryManager()
    action_detector = None

    cap = open_camera()
    print("Face recognition + face locking running. Press 'q' to quit.")

    # Lock state
    locked_name = None
    locked_kps = None
    locked_last_seen = 0.0
    events = []

    # UI: show last action label for 1 second
    last_action = ""
    last_action_time = 0.0
    action_display_duration = 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        vis = frame.copy()
        faces = det.detect(frame)

        # --- Step 1: recognize every detected face ---
        recognized_faces = []

        for f in faces:
            name = "Unknown"
            conf = 0.0
            accepted = False

            aligned, _ = align_face_5pt(frame, f.kps, out_size=ALIGN_SIZE)
            if aligned is not None:
                res = embedder.embed(aligned)
                mr = matcher.match(res.embedding)
                name, accepted = stabilizer.resolve(
                    f.kps,
                    mr.name,
                    mr.similarity,
                    mr.accepted,
                )
                conf = mr.similarity
            else:
                name = "Unknown"
                conf = 0.0
                accepted = False

            recognized_faces.append({
                "face": f,
                "name": name,
                "conf": conf,
                "accepted": accepted,
            })

        detected_person = None

        # --- Step 2: lock / track / unlock logic ---
        if locked_name is None:
            # SEARCHING: wait for target identity to appear
            candidates = [
                r for r in recognized_faces
                if r["accepted"] and r["name"] == target_name
            ]

            if candidates:
                candidates.sort(key=lambda x: x["conf"], reverse=True)
                top = candidates[0]

                locked_name = top["name"]
                locked_last_seen = now
                detected_person = top
                locked_kps = top["face"].kps.copy()

                action_detector = ActionDetector()
                history_manager.log_event(locked_name, "LOCKED")
                write_lock_state(
                    target_name=target_name,
                    locked_name=locked_name,
                    is_locked=True,
                    last_action="LOCKED",
                )
                print(f"[lock] Locked on {locked_name} (sim={top['conf']:.3f})")

        else:
            # LOCKED: prefer identity match, fall back to spatial tracking
            found_by_id = None
            for r in recognized_faces:
                if r["name"] == locked_name:
                    if found_by_id is None or r["conf"] > found_by_id["conf"]:
                        found_by_id = r

            if found_by_id:
                detected_person = found_by_id
                locked_last_seen = now
                locked_kps = detected_person["face"].kps.copy()

            elif locked_kps is not None:
                # Spatial fallback: nearest face by 5pt keypoint distance
                best_track = None
                min_dist = 1e9

                for r in recognized_faces:
                    if r["name"] != locked_name and r["name"] != "Unknown":
                        continue

                    dist = np.max(np.linalg.norm(r["face"].kps - locked_kps, axis=1))
                    if dist < TRACKING_THRESHOLD_PX:
                        if dist < min_dist:
                            min_dist = dist
                            best_track = r

                if best_track:
                    detected_person = best_track
                    detected_person["name"] = locked_name
                    detected_person["accepted"] = True
                    locked_last_seen = now
                    locked_kps = detected_person["face"].kps.copy()

            # Release lock if target missing longer than LOCK_GRACE_S
            if detected_person is None and (now - locked_last_seen > LOCK_GRACE_S):
                print(f"[lock] Lost {locked_name}, unlocking")
                history_manager.log_event(locked_name, "UNLOCKED")
                write_lock_state(
                    target_name=target_name,
                    is_locked=False,
                    last_action="UNLOCKED",
                )
                locked_name = None
                action_detector = None
                locked_kps = None

        # --- Step 3: draw overlays and log actions ---
        for r in recognized_faces:
            f = r["face"]
            is_target = detected_person is not None and f is detected_person["face"]

            if is_target:
                if action_detector is None:
                    action_detector = ActionDetector()

                bbox = np.array([f.x1, f.y1, f.x2, f.y2])
                actions = action_detector.update(f.kps, bbox)

                for act in actions:
                    history_manager.log_event(locked_name, act)
                    write_lock_state(
                        target_name=target_name,
                        locked_name=locked_name,
                        is_locked=True,
                        last_action=act,
                    )
                    print(f"[event] {locked_name}: {act}")
                    events.append((now, locked_name, act))
                    last_action = act
                    last_action_time = now

                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 230, 255), 2)

                label = f"{locked_name}"
                if now - last_action_time < action_display_duration:
                    label += f" [{last_action}]"

                cv2.putText(
                    vis, label, (f.x1, f.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2,
                )
            else:
                color = (0, 255, 0) if r["accepted"] else (0, 0, 255)
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 1)
                cv2.putText(
                    vis, r["name"], (f.x1, f.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                )

        cv2.imshow("Face Locking System", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
