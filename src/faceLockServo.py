"""
Single-speaker face locking with MQTT motor control (BENAX assessment).

Locks onto one enrolled speaker, ignores other faces, and publishes motor
servo angles (0-180 degrees) and SCAN while searching
to an ESP8266 via MQTT. Uses MediaPipe FaceMesh for detection.

Logs speaker ID, confidence, timestamps, and motor commands to
history_log.jsonl and lock_state.json for the live HTML dashboard.

Run:  python -m src.faceLockServo
      python -m src.dashboard   # second terminal — live event view
Quit: q

MQTT topic / broker: see src/config.py
Firmware: src/servo_controller/servo_controller.ino (servo on D5 / GPIO14)
"""
import sys
import time
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
    MQTT_TOPIC_MOTOR_CMD,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    PREVIEW_WINDOW,
    open_camera,
    scale_preview,
)
from src.embed import ArcFaceEmbedderONNX, EmbeddingResult
from src.history_manager import HistoryManager
from src.lock_state import write_lock_state
from src.motion_control import (
    CMD_SCAN,
    ServoAngleTracker,
    format_angle_value,
    format_motor_display,
)

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

class MQTTMotorController:
    """MQTT publisher for assessment motor commands."""

    def __init__(self, broker=MQTT_BROKER, port=MQTT_PORT):
        self.broker = broker
        self.port = port
        self.topic = MQTT_TOPIC_MOTOR_CMD
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

    def send_command(self, command: str) -> bool:
        """Publish a motor command string via MQTT."""
        if not self.connected:
            return False
        try:
            result = self.client.publish(self.topic, command, qos=1)
            result.wait_for_publish(timeout=0.25)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"[MQTT] Error sending command: {e}")
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
        self.last_motor_command = ""
        self.last_confidence = 0.0
        self._target_was_visible = False

        write_lock_state(target_name=target_name, is_locked=False)
        
        # Initialize MQTT motor controller
        self.motor_controller = MQTTMotorController()
        if not self.motor_controller.connect():
            print(
                "[MQTT] WARNING: Broker not connected — servo will NOT move until "
                f"{MQTT_BROKER}:{MQTT_PORT} is reachable."
            )

        # MediaPipe detector and embedder
        self.detector = MediaPipeFaceDetector(min_size=(35, 35))
        self.embedder = ArcFaceEmbedderONNX(
            model_path=str(MODEL_PATH),
            debug=False,
        )
        
        # Face position -> servo angle tracker
        self.angle_tracker: Optional[ServoAngleTracker] = None
        
        print(f"[MQTT] Ready to send servo angles for {target_name}")

    def _publish_motor_command(
        self,
        command: str,
        *,
        confidence: Optional[float] = None,
        log_tracking: bool = True,
        always_send: bool = False,
    ) -> None:
        if not command:
            return
        if (
            not always_send
            and command.isdigit()
            and command == getattr(self, "_last_mqtt_command", "")
        ):
            self.last_motor_command = command
            return
        sent = self.motor_controller.send_command(command)
        if command != getattr(self, "_last_mqtt_command", ""):
            self._last_mqtt_command = command
            if sent:
                status = "ok"
            elif not self.motor_controller.connected:
                status = "FAILED (MQTT not connected — ESP8266 will not move)"
            else:
                status = "FAILED (publish timeout)"
            print(f"[MOTOR] -> {format_motor_display(command)} ({status})")
        self.last_motor_command = command
        if confidence is not None:
            self.last_confidence = confidence
        if log_tracking and command != getattr(self, "_last_logged_command", ""):
            self._last_logged_command = command
            conf = confidence if confidence is not None else self.last_confidence
            self.history_manager.log_tracking(self.target_name, conf, command)
        write_lock_state(
            target_name=self.target_name,
            locked_name=self.target_name if self.locked else None,
            is_locked=self.locked,
            last_action=self.last_action or command,
            confidence=self.last_confidence,
            motor_command=command,
        )

    def _start_search(self, reason: str = "") -> None:
        """Enter SCAN mode — servo hunts until the target is seen again."""
        if not self.searching:
            suffix = f" ({reason})" if reason else ""
            print(f"[SEARCH] SCAN mode to find {self.target_name.upper()}{suffix}")
        self.searching = True

    def _stop_search(self) -> None:
        """Leave search mode — tracking resumes on the locked person."""
        if self.searching:
            print(f"[SEARCH] Target found — tracking {self.target_name.upper()}")
        self.searching = False

    def on_locked(
        self,
        face_center_x: Optional[float] = None,
        confidence: float = 0.0,
    ) -> None:
        """Record lock event, stop search, and publish initial motor command."""
        self._stop_search()
        self.action_detector = ActionDetector()
        self.last_confidence = confidence
        self.history_manager.log_event(
            self.target_name, "LOCKED", confidence=confidence
        )
        write_lock_state(
            target_name=self.target_name,
            locked_name=self.target_name,
            is_locked=True,
            last_action="LOCKED",
            confidence=confidence,
        )

        if self.angle_tracker is not None and face_center_x is not None:
            cmd = self.angle_tracker.snap_to_face(face_center_x)
            self._publish_motor_command(
                cmd,
                confidence=confidence,
                log_tracking=False,
                always_send=True,
            )

        print(f"[LOCKED] Target locked — dashboard updated")

    def on_unlocked(self) -> None:
        """Record unlock event for dashboard + history and start SCAN."""
        self.history_manager.log_event(self.target_name, "UNLOCKED")
        write_lock_state(
            target_name=self.target_name,
            is_locked=False,
            last_action="UNLOCKED",
            motor_command=CMD_SCAN,
        )
        self.action_detector = None
        self._start_search("target left frame")
        self._publish_motor_command(CMD_SCAN, log_tracking=True)
        print(f"[LOST] Lost signal on {self.target_name.upper()} — dashboard updated")

    def run_search_sweep(self) -> Optional[str]:
        """Publish SCAN while searching for the enrolled speaker."""
        if self.angle_tracker is None:
            return None
        cmd = self.angle_tracker.update(None, searching=True)
        self._publish_motor_command(cmd, log_tracking=True)
        return cmd

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
        """Initialize servo angle tracker with frame width."""
        self.angle_tracker = ServoAngleTracker(width)
    
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
    
    def find_target_face(
        self, recognized_faces: List[Tuple[dict, str, float]]
    ) -> Tuple[Optional[dict], float, float]:
        """Find the enrolled speaker among detected faces (ignore others)."""
        for face_data, identity, similarity in recognized_faces:
            if identity == self.target_name:
                distance = 1.0 - similarity
                return face_data, distance, similarity
        
        return None, 1.0, 0.0
    
    def update_motion_tracking(
        self,
        target_face_data: Optional[Dict],
        confidence: float,
        *,
        force: bool = False,
    ) -> Optional[str]:
        """Update tracking error and publish motor command when appropriate."""
        if not self.angle_tracker:
            return None

        face_center_x = (
            target_face_data.get("center_x") if target_face_data else None
        )
        cmd = self.angle_tracker.update(
            face_center_x,
            searching=False,
            force=force,
        )
        if cmd is not None:
            self._publish_motor_command(cmd, confidence=confidence)
        else:
            write_lock_state(
                target_name=self.target_name,
                locked_name=self.target_name,
                is_locked=True,
                last_action=self.last_action or "TRACKING",
                confidence=confidence,
                motor_command=str(int(round(self.angle_tracker.current_angle))),
            )
        return str(int(round(self.angle_tracker.current_angle)))
    
    def close(self):
        """Close the face locker and MQTT controller."""
        self.motor_controller.disconnect()

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
    locker._start_search("waiting for target")
    locker.run_search_sweep()

    preview_w = PREVIEW_WIDTH if PREVIEW_WIDTH > 0 else width
    preview_h = PREVIEW_HEIGHT if PREVIEW_HEIGHT > 0 else height
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, preview_w, preview_h)

    print(f"\n" + "="*50)
    print(f"Single-Speaker Face Locking + MQTT Motor Control")
    print(f"Target: {choice.upper()}")
    print(f"Camera: {width}x{height} (native preview)")
    print(f"MQTT Topic: {MQTT_TOPIC_MOTOR_CMD}")
    print(f"Tracking: face position -> servo angle 0-180 (SCAN while searching)")
    print(f"Dashboard: run python -m src.dashboard (innovation — live evidence view)")
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
            
            # Find target face (single-identity speaker lock)
            target_face, target_distance, target_confidence = locker.find_target_face(
                recognized_faces
            )

            motor_cmd = None
            target_visible = (
                target_face is not None and target_distance <= DISTANCE_THRESHOLD
            )

            if not locker.locked:
                # Not locked yet: SCAN and try to acquire the enrolled speaker
                motor_cmd = locker.run_search_sweep()

                if target_visible:
                    locker.locked = True
                    locker.fail_count = 0
                    locker.total_lock_frames = 0
                    locker.on_locked(target_face["center_x"], target_confidence)
                    print(
                        f"[LOCKED] {locker.target_name.upper()} "
                        f"conf={target_confidence:.2f} at "
                        f"({int(target_face['center_x'])}, {int(target_face['center_y'])})"
                    )

            elif target_visible:
                # LOCKED + speaker in frame: servo follows face horizontally
                reacquired = locker.locked and not locker._target_was_visible
                locker._stop_search()
                if reacquired and locker.angle_tracker:
                    snap = locker.angle_tracker.snap_to_face(
                        target_face["center_x"]
                    )
                    locker._publish_motor_command(
                        snap,
                        confidence=target_confidence,
                        always_send=True,
                    )
                    print(
                        f"[REACQUIRED] {locker.target_name.upper()} "
                        f"— servo snapped to {snap}°"
                    )
                motor_cmd = locker.update_motion_tracking(
                    target_face,
                    target_confidence,
                )
                locker.fail_count = 0
                locker.total_lock_frames += 1
                face = target_face["face_data"]

                locker.process_locked_actions(face)

                cv2.rectangle(vis, (face["x1"], face["y1"]), (face["x2"], face["y2"]), (0, 255, 0), 2)

                label = f"LOCKED: {locker.target_name.upper()} ({target_confidence:.2f})"
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

                angle_val = (
                    float(motor_cmd)
                    if motor_cmd and motor_cmd.isdigit()
                    else locker.angle_tracker.current_angle
                    if locker.angle_tracker
                    else 90.0
                )
                angle_text = format_angle_value(angle_val)
                cv2.putText(
                    vis,
                    f"Servo: {angle_text}",
                    (face["x1"], face["y2"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                if frames % 10 == 0 and locker.angle_tracker:
                    dbg = locker.angle_tracker.debug_line()
                    print(
                        f"[TRACKING] {locker.target_name.upper()} | "
                        f"conf={target_confidence:.2f} | {dbg}"
                    )

            else:
                # LOCKED but occluded / out of frame — SCAN sweep to re-acquire
                locker._start_search("target out of frame")
                motor_cmd = locker.run_search_sweep()
                locker.fail_count += 1

                if locker.fail_count >= LOCK_RELEASE_FRAMES:
                    locker.locked = False
                    locker.fail_count = 0
                    locker.on_unlocked()

            locker._target_was_visible = target_visible
            
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
                locked_status = "SCAN (re-acquire)..."
            else:
                locked_status = "SCAN..."
            if motor_cmd == CMD_SCAN:
                cmd_display = "SCAN"
            elif motor_cmd and motor_cmd.isdigit():
                cmd_display = format_angle_value(float(motor_cmd))
            elif locker.angle_tracker:
                cmd_display = format_angle_value(locker.angle_tracker.current_angle)
            else:
                cmd_display = format_motor_display(
                    locker.last_motor_command or ""
                )
            if cmd_display:
                status = (
                    f"Target: {locker.target_name} | {locked_status} | "
                    f"Angle: {cmd_display} | FPS: {fps:.1f}"
                )
            else:
                status = f"Target: {locker.target_name} | {locked_status} | FPS: {fps:.1f}"
            cv2.putText(vis, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(vis, "Press 'q' to quit", (10, H - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow(PREVIEW_WINDOW, scale_preview(vis))
            
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