# src/detect.py
import cv2
import time
import numpy as np
from pathlib import Path

from .recognize import align_face_5pt, ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz, HaarFaceMesh5pt
from .action_detector import ActionDetector
from .history_manager import HistoryManager


def main():
    db_path = Path("data/db/face_db.npz")
    det = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(load_db_npz(db_path), dist_thresh=0.34)
    
    # History and Actions
    history_manager = HistoryManager()
    action_detector = None # Will instantiate when locked

    cap = None
    for _idx in (0, 1, 2):
        _cap = cv2.VideoCapture(_idx)
        if _cap.isOpened():
            cap = _cap
            print(f"Camera opened on index {_idx}.")
            break
        _cap.release()
    if cap is None:
        raise RuntimeError("Camera not opened. Tried indices 0, 1, 2.")

    print("Face recognition + face locking running. Press 'q' to quit.")

    # Face-locking state
    locked_name = None
    locked_kps = None
    locked_last_seen = 0.0
    lock_grace_s = 2.0
    focus_face = None # The Face object that is currently locked onto
    events = [] # To store logged events
    
    # UI State
    last_action = ""
    last_action_time = 0.0
    action_display_duration = 1.0 # Show action for 1 second

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        vis = frame.copy()

        faces = det.detect(frame)

        # 1. Analyze ALL faces first
        # We will separate them into "The Locked One" vs "Others"
        recognized_faces = []
        
        for f in faces:
            name = "Unknown"
            conf = 0.0
            accepted = False
            
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            if aligned is not None:
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)
                if mr.accepted:
                    name = mr.name
                    conf = mr.similarity
                    accepted = True
                else:
                    conf = mr.similarity # even if unknown
            
            recognized_faces.append({
                "face": f,
                "name": name,
                "conf": conf,
                "accepted": accepted
            })

        # 2. Logic: Determine lock state
        detected_person = None  # The actual face object for the locked target
        
        if locked_name is None:
            # SEARCHING: Look for a strong match to lock onto
            best_sim = -1.0
            # Filter candidates: Must be accepted AND named "Hope"
            candidates = [r for r in recognized_faces if r["accepted"] and r["name"] == "Kheira"]
            
            if candidates:
                # Pick best match
                candidates.sort(key=lambda x: x["conf"], reverse=True)
                top = candidates[0]
                
                locked_name = top["name"]
                locked_last_seen = now
                detected_person = top
                locked_kps = top["face"].kps.copy()
                
                action_detector = ActionDetector()
                history_manager.log_event(locked_name, "LOCKED")
                print(f"[lock] Locked on {locked_name} (sim={top['conf']:.3f})")

        else:
            # LOCKED: Find the specific person
            
            # First, look for identity match (strongest signal)
            found_by_id = None
            for r in recognized_faces:
                if r["name"] == locked_name:
                    if found_by_id is None or r["conf"] > found_by_id["conf"]:
                        found_by_id = r
            
            if found_by_id:
                detected_person = found_by_id
                locked_last_seen = now
                locked_kps = detected_person["face"].kps.copy()
            
            # Fallback: Tracking (Spatial continuity) if identity flickers (e.g. partial occlusion)
            # Only if no ID match found, and we have a face close to last position that isn't positively identified as SOMEONE ELSE
            elif locked_kps is not None:
                best_track = None
                min_dist = 1e9
                
                for r in recognized_faces:
                    # Don't track if it's confidently someone else
                    if r["accepted"] and r["name"] != locked_name:
                        continue
                        
                    dist = np.max(np.linalg.norm(r["face"].kps - locked_kps, axis=1))
                    # Tracking threshold
                    if dist < 50.0: # pixels
                        if dist < min_dist:
                            min_dist = dist
                            best_track = r
                
                if best_track:
                    detected_person = best_track
                    locked_last_seen = now
                    locked_kps = detected_person["face"].kps.copy()

            # Check for lock expiry
            if detected_person is None and (now - locked_last_seen > lock_grace_s):
                print(f"[lock] Lost {locked_name}, unlocking")
                locked_name = None
                action_detector = None
                locked_kps = None
        
        # 3. Draw Everything
        for r in recognized_faces:
            f = r["face"]
            
            # Is this the locked person?
            is_target = (detected_person is not None and f is detected_person["face"])
            
            if is_target:
                # --- DRAW LOCKED TARGET (Yellow) ---
                if action_detector is None: action_detector = ActionDetector()
                
                bbox = np.array([f.x1, f.y1, f.x2, f.y2])
                actions = action_detector.update(f.kps, bbox)
                
                for act in actions:
                    history_manager.log_event(locked_name, act)
                    print(f"[event] {locked_name}: {act}")
                    events.append((now, locked_name, act))
                    last_action = act
                    last_action_time = now

                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 230, 255), 2)
                
                label = f"{locked_name}"
                if now - last_action_time < action_display_duration:
                    label += f" [{last_action}]"
                
                cv2.putText(vis, label, (f.x1, f.y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
            
            else:
                # --- DRAW OTHERS (Green/Red) ---
                # "Other faces are recognised as they are"
                # If accepted -> Green, If Unknown -> Red/Dim
                color = (0, 255, 0) if r["accepted"] else (0, 0, 255)
                name_lbl = r["name"]
                
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 1)
                
                # Show name 
                cv2.putText(vis, f"{name_lbl}", (f.x1, f.y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow("Face Locking System", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
