"""
Face locking with MQTT servo control.

Locks onto an enrolled identity and publishes horizontal servo angles (0-180)
to an ESP8266 via MQTT. Uses full MediaPipe FaceMesh (468 landmarks) for
detection in this mode (different from detect.py which uses HaarFaceMesh5pt).

Also logs LOCKED, movements, smile, and blink to history_log.jsonl and
lock_state.json (same as detect.py) for the HTML dashboard.

Run:  python -m src.faceLockServo
      python -m src.dashboard   # second terminal — live event view
Quit: q

MQTT topic / broker: see src/config.py
Firmware: src/servo_controller/servo_controller.ino
"""
import sys
import time
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
import cv2
import numpy as np
import mediapipe as mp
import paho.mqtt.client as mqtt

from src.action_detector import ActionDetector
from src.config import (
    DB_PATH,
    DISTANCE_THRESHOLD,
    LOCK_RELEASE_FRAMES,
    MATCH_MARGIN,
    MIN_DETECTION_CONFIDENCE,
    MIN_RECOGNITION_SIMILARITY,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC_SERVO_ANGLE,
    SERVO_ANGLE_MAX,
    SERVO_ANGLE_MIN,
    SERVO_DEADZONE_DEG,
    SERVO_INVERTED,
    SERVO_PUBLISH_INTERVAL_MS,
    SERVO_SEARCH_INTERVAL_MS,
    SERVO_SEARCH_STEP_DEG,
    SERVO_SMOOTHING_FACTOR,
    SERVO_TRACK_HEARTBEAT_MS,
    open_camera,
)
from src.embed import ArcFaceEmbedderONNX, EmbeddingResult
from src.history_manager import HistoryManager
from src.lock_state import write_lock_state

# MediaPipe FaceMesh landmark indices (full 468-point mesh)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 291, 0, 17]

# 5-point subset used for ArcFace alignment before embedding
LEFT_EYE_CENTER_IDX = 33
RIGHT_EYE_CENTER_IDX = 263
NOSE_IDX = 1
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291

# -------------------------
# MQTT Controller
# -------------------------

class MQTTServoController:
    """Simple MQTT publisher for servo control."""

    def __init__(self, broker=MQTT_BROKER, port=MQTT_PORT):
        self.broker = broker
        self.port = port
        self.topic = MQTT_TOPIC_SERVO_ANGLE
        self.connected = False

        # Support paho-mqtt v1 and v2
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            self.client = mqtt.Client()  # paho-mqtt < 2.0

        # on_connect fires when the broker sends CONNACK
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"[MQTT] Connected to broker at {self.broker}:{self.port}")
        else:
            self.connected = False
            print(f"[MQTT] Connection refused — return code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect (rc={rc}), will auto-reconnect...")

    def connect(self):
        """Connect to MQTT broker (non-blocking, uses background loop)."""
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()  # background thread handles reconnects
            # Wait up to 3 s for the broker to confirm
            for _ in range(30):
                if self.connected:
                    return True
                time.sleep(0.1)
            print(f"[MQTT] Timed out waiting for broker at {self.broker}:{self.port}")
            return False
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}")
            return False

    def send_angle(self, angle):
        """Send servo angle via MQTT."""
        if not self.connected:
            return False
        try:
            angle = max(0, min(180, int(angle)))
            result = self.client.publish(self.topic, str(angle), qos=1)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"[MQTT] Error sending angle: {e}")
            return False

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            print("[MQTT] Disconnected from broker")

# -------------------------
# Database helpers
# -------------------------

def load_database() -> Dict[str, np.ndarray]:
    """Load enrolled face database."""
    if not DB_PATH.exists():
        return {}
    data = np.load(str(DB_PATH), allow_pickle=True)
    db = {}
    for k in data.files:
        emb = np.asarray(data[k], dtype=np.float32)
        if emb.ndim > 1:
            emb = emb.reshape(-1)
        db[k] = emb
    return db

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b))

def recognize_face(embedding: np.ndarray, db: Dict[str, np.ndarray], threshold: float) -> Tuple[str, float]:
    """Recognize a face by comparing embedding with database."""
    best_name = "Unknown"
    best_similarity = 0.0
    second_similarity = 0.0

    for name, ref_emb in db.items():
        similarity = cosine_similarity(embedding, ref_emb)
        if similarity > best_similarity:
            second_similarity = best_similarity
            best_similarity = similarity
            best_name = name
        elif similarity > second_similarity:
            second_similarity = similarity

    margin = best_similarity - second_similarity
    accepted = (
        best_similarity >= (1.0 - threshold)
        or best_similarity >= MIN_RECOGNITION_SIMILARITY
        or (margin >= MATCH_MARGIN and best_similarity >= MIN_RECOGNITION_SIMILARITY - 0.05)
    )
    if not accepted:
        return "Unknown", best_similarity
    return best_name, best_similarity

# -------------------------
# Position Tracking
# -------------------------

class PositionTracker:
    """Tracks face position and converts to servo angles."""

    def __init__(self, screen_width: int):
        self.screen_width = screen_width
        self.current_angle = 90.0
        self.position_buffer = []
        self.buffer_size = 5
        # Rate-limiting: track the last time we published
        self.last_publish_time = 0.0
        self.last_sent_angle = 90.0
        self.last_heartbeat_time = 0.0

    def calculate_angle_from_position(self, face_center_x: float) -> float:
        """Calculate servo angle from face X position on screen.

        Face on LEFT  → normalized_x near 0 → angle near 180°
        Face on RIGHT → normalized_x near 1 → angle near 0°
        Flip SERVO_INVERTED=True if your motor goes the wrong way.
        """
        normalized_x = max(0.0, min(1.0, face_center_x / self.screen_width))
        if SERVO_INVERTED:
            angle = normalized_x * SERVO_ANGLE_MAX          # 0 left → 180 right
        else:
            angle = SERVO_ANGLE_MAX - (normalized_x * SERVO_ANGLE_MAX)  # 180 left → 0 right

        # Rolling average to smooth out landmark jitter
        self.position_buffer.append(angle)
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
        return float(np.mean(self.position_buffer))

    def sync_angle(self, angle: float) -> None:
        """Seed tracker state after search/re-lock so tracking starts immediately."""
        angle = float(max(SERVO_ANGLE_MIN, min(SERVO_ANGLE_MAX, angle)))
        self.current_angle = angle
        self.last_sent_angle = angle
        self.position_buffer = [angle]
        now_ms = time.time() * 1000
        self.last_publish_time = now_ms
        self.last_heartbeat_time = now_ms

    def update(self, face_center_x: Optional[float], *, force: bool = False) -> Optional[float]:
        """Smooth the angle and return the value to publish (or None if suppressed)."""
        if face_center_x is None:
            return None

        target_angle = self.calculate_angle_from_position(face_center_x)
        # Exponential smoothing toward the target
        self.current_angle += (target_angle - self.current_angle) * SERVO_SMOOTHING_FACTOR

        now_ms = time.time() * 1000

        if force:
            self.last_publish_time = now_ms
            self.last_heartbeat_time = now_ms
            self.last_sent_angle = self.current_angle
            return self.current_angle

        moved_enough = abs(self.current_angle - self.last_sent_angle) >= SERVO_DEADZONE_DEG
        rate_ok = now_ms - self.last_publish_time >= SERVO_PUBLISH_INTERVAL_MS
        heartbeat_due = now_ms - self.last_heartbeat_time >= SERVO_TRACK_HEARTBEAT_MS

        if not moved_enough and not heartbeat_due:
            return None
        if moved_enough and not rate_ok:
            return None

        self.last_publish_time = now_ms
        self.last_heartbeat_time = now_ms
        self.last_sent_angle = self.current_angle
        return self.current_angle


class ServoSearchController:
    """Sweep servo while the target is not visible (search mode)."""

    def __init__(
        self,
        angle_min: int = SERVO_ANGLE_MIN,
        angle_max: int = SERVO_ANGLE_MAX,
        step_deg: int = SERVO_SEARCH_STEP_DEG,
        interval_ms: int = SERVO_SEARCH_INTERVAL_MS,
    ):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.step_deg = step_deg
        self.interval_ms = interval_ms
        self.current_angle = 90.0
        self.direction = 1
        self.last_publish_time = 0.0

    def sync_from(self, angle: float) -> None:
        """Align sweep position with the last tracking angle."""
        self.current_angle = float(max(self.angle_min, min(self.angle_max, angle)))

    def update(self) -> Optional[float]:
        """Return the next sweep angle to publish, or None if rate-limited."""
        now_ms = time.time() * 1000
        if now_ms - self.last_publish_time < self.interval_ms:
            return None

        self.current_angle += self.step_deg * self.direction
        if self.current_angle >= self.angle_max:
            self.current_angle = float(self.angle_max)
            self.direction = -1
        elif self.current_angle <= self.angle_min:
            self.current_angle = float(self.angle_min)
            self.direction = 1

        self.last_publish_time = now_ms
        return self.current_angle

# -------------------------
# MediaPipe Face Detector
# -------------------------

class MediaPipeFaceDetector:
    """Face detector using MediaPipe Face Mesh."""
    
    def __init__(self, min_size=(50, 50)):
        self.min_size = min_size
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
    
    def detect(self, frame: np.ndarray, max_faces: int = 5) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe and return face data."""
        faces = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append([lm.x * w, lm.y * h, lm.z * w])
                landmarks = np.array(landmarks)
                
                x_min = int(np.min(landmarks[:, 0]))
                y_min = int(np.min(landmarks[:, 1]))
                x_max = int(np.max(landmarks[:, 0]))
                y_max = int(np.max(landmarks[:, 1]))
                
                width = x_max - x_min
                height = y_max - y_min
                if width < self.min_size[0] or height < self.min_size[1]:
                    continue
                
                face_data = {
                    'x1': x_min, 'y1': y_min, 'x2': x_max, 'y2': y_max,
                    'landmarks': landmarks,
                    'width': width, 'height': height
                }
                faces.append(face_data)
                
                if len(faces) >= max_faces:
                    break
        
        return faces

def align_face_mediapipe(frame: np.ndarray, landmarks: np.ndarray, out_size: Tuple[int, int] = (112, 112)):
    """Align face using MediaPipe landmarks for 5-point alignment."""
    if landmarks is None or len(landmarks) < 5:
        return None
    
    src_points = np.array([
        landmarks[LEFT_EYE_CENTER_IDX][:2],
        landmarks[RIGHT_EYE_CENTER_IDX][:2],
        landmarks[NOSE_IDX][:2],
        landmarks[MOUTH_LEFT_IDX][:2],
        landmarks[MOUTH_RIGHT_IDX][:2]
    ], dtype=np.float32)
    
    dst_points = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    
    transform = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.RANSAC)[0]
    if transform is None:
        return None
    
    aligned_face = cv2.warpAffine(frame, transform, out_size, borderValue=0.0)
    return aligned_face


def landmarks_to_5pt(landmarks: np.ndarray) -> np.ndarray:
    """Extract ArcFace-style 5-point keypoints from MediaPipe landmarks."""
    return np.array(
        [
            landmarks[LEFT_EYE_CENTER_IDX][:2],
            landmarks[RIGHT_EYE_CENTER_IDX][:2],
            landmarks[NOSE_IDX][:2],
            landmarks[MOUTH_LEFT_IDX][:2],
            landmarks[MOUTH_RIGHT_IDX][:2],
        ],
        dtype=np.float32,
    )

# -------------------------
# Face Locker
# -------------------------

class FaceLocker:
    """Locks onto a specific face and tracks it with servo control."""
    
    def __init__(self, target_name: str, target_embedding: np.ndarray, db: Dict[str, np.ndarray]):
        self.target_name = target_name
        self.target_embedding = target_embedding
        self.db = db
        
        self.locked = False
        self.searching = True
        self.fail_count = 0
        self.total_lock_frames = 0
        self.action_detector: Optional[ActionDetector] = None
        self.history_manager = HistoryManager()
        self.last_action = ""
        self.last_action_time = 0.0
        self.action_display_duration = 1.0

        write_lock_state(target_name=target_name, is_locked=False)
        
        # Initialize MQTT servo controller
        self.servo_controller = MQTTServoController()
        self.servo_controller.connect()
        
        # MediaPipe detector and embedder
        self.detector = MediaPipeFaceDetector(min_size=(50, 50))
        self.embedder = ArcFaceEmbedderONNX(
            model_path=str(MODEL_PATH),
            debug=False,
        )
        
        # Position tracker + search sweep (used when target leaves the frame)
        self.position_tracker = None
        self.search_controller = ServoSearchController()
        
        print(f"[MQTT] Ready to send servo commands for {target_name}")

    def _publish_servo_angle(self, angle: float) -> None:
        if angle is not None:
            self.servo_controller.send_angle(angle)

    def _start_search(self, reason: str = "") -> None:
        """Enter sweep mode — servo hunts until the target is seen again."""
        if not self.searching:
            suffix = f" ({reason})" if reason else ""
            print(f"[SEARCH] Servo sweeping to find {self.target_name.upper()}{suffix}")
        self.searching = True
        if self.position_tracker is not None:
            self.search_controller.sync_from(self.position_tracker.current_angle)

    def _stop_search(self) -> None:
        """Leave sweep mode — tracking resumes on the locked person."""
        if self.searching:
            print(f"[SEARCH] Target found — servo tracking {self.target_name.upper()}")
        self.searching = False

    def on_locked(self, face_center_x: Optional[float] = None) -> None:
        """Record lock event, stop search, and snap servo to the face."""
        self._stop_search()
        self.action_detector = ActionDetector()
        self.history_manager.log_event(self.target_name, "LOCKED")
        write_lock_state(
            target_name=self.target_name,
            locked_name=self.target_name,
            is_locked=True,
            last_action="LOCKED",
        )

        if self.position_tracker is not None and face_center_x is not None:
            track_angle = self.position_tracker.update(face_center_x, force=True)
            if track_angle is not None:
                self._publish_servo_angle(track_angle)
                self.search_controller.sync_from(track_angle)

        print(f"[LOCKED] Target locked — dashboard updated")

    def on_unlocked(self) -> None:
        """Record unlock event for dashboard + history and start servo search."""
        self.history_manager.log_event(self.target_name, "UNLOCKED")
        write_lock_state(
            target_name=self.target_name,
            is_locked=False,
            last_action="UNLOCKED",
        )
        self.action_detector = None
        self._start_search("target left frame")
        print(f"[LOST] Lost signal on {self.target_name.upper()} — dashboard updated")

    def run_search_sweep(self) -> Optional[float]:
        """Publish the next sweep angle while searching."""
        sweep_angle = self.search_controller.update()
        if sweep_angle is not None:
            self._publish_servo_angle(sweep_angle)
        return sweep_angle

    def process_locked_actions(self, face_data: Dict[str, Any]) -> List[str]:
        """Detect smile, blink, and head movement on the locked face."""
        if self.action_detector is None:
            self.action_detector = ActionDetector()

        landmarks = face_data["landmarks"]
        kps = landmarks_to_5pt(landmarks)
        bbox = np.array(
            [face_data["x1"], face_data["y1"], face_data["x2"], face_data["y2"]],
            dtype=np.float32,
        )
        actions = self.action_detector.update(kps, bbox)

        for act in actions:
            self.history_manager.log_event(self.target_name, act)
            write_lock_state(
                target_name=self.target_name,
                locked_name=self.target_name,
                is_locked=True,
                last_action=act,
            )
            self.last_action = act
            self.last_action_time = time.time()
            print(f"[event] {self.target_name}: {act}")

        return actions
    
    def set_screen_size(self, width: int):
        """Initialize position tracker with screen width."""
        self.position_tracker = PositionTracker(width)
    
    def recognize_all_faces(self, frame: np.ndarray, faces) -> List[Tuple[dict, str, float]]:
        """Recognize all faces in the frame."""
        results = []
        
        for face_data in faces:
            aligned = align_face_mediapipe(frame, face_data['landmarks'])
            if aligned is None:
                continue
            
            res: EmbeddingResult = self.embedder.embed(aligned)
            identity, similarity = recognize_face(res.embedding, self.db, DISTANCE_THRESHOLD)
            
            center_x = (face_data['x1'] + face_data['x2']) / 2.0
            center_y = (face_data['y1'] + face_data['y2']) / 2.0
            
            face_data_full = {
                'face_data': face_data,
                'center_x': center_x,
                'center_y': center_y,
                'width': face_data['width'],
                'height': face_data['height']
            }
            
            results.append((face_data_full, identity, similarity))
        
        return results
    
    def find_target_face(self, recognized_faces: List[Tuple[dict, str, float]]) -> Tuple[Optional[dict], float]:
        """Find the target face among recognized faces."""
        for face_data, identity, similarity in recognized_faces:
            if identity == self.target_name:
                distance = 1.0 - similarity
                return face_data, distance
        
        return None, 1.0
    
    def update_position_tracking(
        self,
        target_face_data: Optional[Dict],
        *,
        force: bool = False,
    ) -> Optional[float]:
        """Update position tracking and publish servo angle when appropriate."""
        if not self.position_tracker or not target_face_data:
            return None

        face_center_x = target_face_data.get("center_x")
        publish_angle = self.position_tracker.update(face_center_x, force=force)

        if publish_angle is not None:
            self._publish_servo_angle(publish_angle)

        return self.position_tracker.current_angle
    
    def close(self):
        """Close the face locker and MQTT controller."""
        self.servo_controller.disconnect()

# -------------------------
# Main function
# -------------------------

def main():
    """Enhanced Face Locking main function with MQTT servo control."""
    
    # Load database
    db = load_database()
    if not db:
        print("ERROR: No enrolled identities. Run: python -m src.enroll")
        return False
    
    # Show available identities
    names = sorted(db.keys())
    print("\nEnrolled identities:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    
    # Let user choose target
    print("\nEnter the name of the identity to lock (exact match): ", end="")
    try:
        choice = input().strip()
    except EOFError:
        choice = names[0] if names else ""
    
    if not choice:
        choice = names[0] if names else ""
    
    if choice not in db:
        print(f"ERROR: '{choice}' not in database. Choose from: {names}")
        return False
    
    print(f"Will lock onto: {choice}")
    
    # Initialize face locker
    locker = FaceLocker(choice, db[choice], db)
    
    try:
        cap = open_camera()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return False
    
    # Get screen dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    locker.set_screen_size(width)
    
    print(f"\n" + "="*50)
    print(f"Face Locking with Servo Control")
    print(f"Target: {choice.upper()}")
    print(f"Screen: {width}x{height}")
    print(f"Servo Range: {SERVO_ANGLE_MIN}° to {SERVO_ANGLE_MAX}°")
    print(f"MQTT Topic: {MQTT_TOPIC_SERVO_ANGLE}")
    print(f"Dashboard: run python -m src.dashboard (logs movements, smile, blink)")
    print(f"Press 'q' to quit")
    print("="*50 + "\n")
    
    # Performance tracking
    t0 = time.time()
    frames = 0
    fps = 0.0
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()
            
            H, W = frame.shape[:2]
            vis = frame.copy()
            
            # Detect faces
            faces = locker.detector.detect(frame, max_faces=5)
            
            # Recognize faces
            recognized_faces = locker.recognize_all_faces(frame, faces)
            
            # Find target face
            target_face, target_distance = locker.find_target_face(recognized_faces)

            # servo_angle is used for the on-screen display only
            servo_angle = None

            target_visible = (
                target_face is not None and target_distance <= DISTANCE_THRESHOLD
            )

            if not locker.locked:
                # Not locked yet: sweep servo and try to acquire the target
                servo_angle = locker.run_search_sweep()

                if target_visible:
                    locker.locked = True
                    locker.fail_count = 0
                    locker.total_lock_frames = 0
                    locker.on_locked(target_face["center_x"])
                    print(
                        f"[LOCKED] Target locked at "
                        f"({int(target_face['center_x'])}, {int(target_face['center_y'])})"
                    )

            elif target_visible:
                # LOCKED + face in frame: track target, stop any search sweep
                locker._stop_search()
                servo_angle = locker.update_position_tracking(target_face)
                locker.fail_count = 0
                locker.total_lock_frames += 1
                face = target_face["face_data"]

                locker.process_locked_actions(face)

                cv2.rectangle(vis, (face["x1"], face["y1"]), (face["x2"], face["y2"]), (0, 255, 0), 2)

                center_x = (face["x1"] + face["x2"]) // 2
                center_y = (face["y1"] + face["y2"]) // 2
                cv2.line(vis, (center_x, face["y1"]), (center_x, face["y2"]), (0, 255, 0), 1)
                cv2.line(vis, (face["x1"], center_y), (face["x2"], center_y), (0, 255, 0), 1)

                label = f"LOCKED: {locker.target_name.upper()}"
                if time.time() - locker.last_action_time < locker.action_display_duration:
                    label += f" [{locker.last_action}]"

                cv2.putText(
                    vis,
                    label,
                    (face["x1"], max(0, face["y1"] - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if servo_angle is not None:
                    angle_text = f"Servo: {servo_angle:.1f}°"
                    cv2.putText(
                        vis,
                        angle_text,
                        (face["x1"], face["y2"] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                if frames % 10 == 0:
                    pos_x, pos_y = int(target_face["center_x"]), int(target_face["center_y"])
                    angle_str = f"{servo_angle:.1f}°" if servo_angle is not None else "N/A"
                    print(
                        f"[TRACKING] {locker.target_name.upper()} | "
                        f"Pos: ({pos_x},{pos_y}) | Angle: {angle_str}"
                    )

            else:
                # LOCKED but face left frame: sweep immediately, unlock after grace period
                locker._start_search("target out of frame")
                servo_angle = locker.run_search_sweep()
                locker.fail_count += 1

                if locker.fail_count >= LOCK_RELEASE_FRAMES:
                    locker.locked = False
                    locker.fail_count = 0
                    locker.on_unlocked()
            
            # Draw other faces
            for face_data, identity, similarity in recognized_faces:
                face = face_data['face_data']
                
                if locker.locked and identity == locker.target_name:
                    continue
                
                if identity == "Unknown":
                    color = (0, 0, 255)
                elif identity == locker.target_name and not locker.locked:
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 0)
                
                cv2.rectangle(vis, (face['x1'], face['y1']), (face['x2'], face['y2']), color, 2)
                
                label = identity if identity != "Unknown" else f"Unknown ({similarity:.2f})"
                cv2.putText(vis, label, 
                           (face['x1'], max(0, face['y1'] - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw status
            if locker.locked and not locker.searching:
                locked_status = "LOCKED"
            elif locker.searching:
                locked_status = "Searching (servo sweep)..."
            else:
                locked_status = "Searching..."
            if servo_angle is not None:
                status = (
                    f"Target: {locker.target_name} | {locked_status} | "
                    f"Angle: {servo_angle:.1f}° | FPS: {fps:.1f}"
                )
            else:
                status = f"Target: {locker.target_name} | {locked_status} | FPS: {fps:.1f}"
            cv2.putText(vis, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(vis, "Press 'q' to quit", (10, H - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow("Face Locking with Servo Control", vis)
            
            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        locker.close()
    
    print("\nFace Locking ended.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)