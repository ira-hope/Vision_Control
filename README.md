# Face Recognition & Face Locking System (Windows)

A **real-time face recognition and face locking system** that runs on **Windows + Python 3.12**.

Pipeline: **Camera → Haar detection → FaceMesh 5-point landmarks → Face alignment (112×112) → ArcFace embedding**

Key capabilities:
- Detect and recognize multiple faces simultaneously
- **Lock onto a specific target face** with identity-based tracking + spatial fallback
- Detect actions on the locked face (head movement, smile)
- Log all events to a `.jsonl` history file
- **Live HTML dashboard** — lock status, movements, and event log in the browser
- Control a **servo motor** via **MQTT over ESP8266** — the servo tracks the locked face's horizontal position

---

## 📁 Project Structure

```
Facelocking2/
│
├── .venv/
├── dashboard/
│   └── index.html               # Live event dashboard (served by dashboard.py)
├── data/
│   ├── db/
│   │   ├── face_db.npz          # Enrolled face embeddings
│   │   └── face_db.json         # DB metadata
│   ├── enroll/                  # Saved crop images per person
│   └── history/
│       ├── history_log.jsonl    # Action event log (JSONL)
│       └── lock_state.json      # Live lock status (written by detect.py)
│
├── models/
│   └── embedder_arcface.onnx    # ArcFace ONNX model
│
├── scripts/
│   └── test_all.ps1             # Automated smoke tests (no camera windows)
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Shared paths, thresholds, MQTT settings
│   ├── detect.py                # Main detection + face locking loop
│   ├── dashboard.py             # HTTP server for live HTML dashboard
│   ├── lock_state.py            # Lock status file read/write for dashboard
│   ├── enroll.py                # Face enrollment tool
│   ├── faceLockServo.py         # Face locking + MQTT servo control
│   ├── haar_5pt.py              # Haar + MediaPipe FaceMesh detector
│   ├── align.py                 # 5-point face alignment
│   ├── embed.py                 # ArcFace ONNX embedder
│   ├── recognize.py             # DB matching / recognition helpers
│   ├── action_detector.py       # Head movement & smile detection
│   ├── history_manager.py       # JSONL event logger + disk reader for dashboard
│   ├── landmarks.py
│   ├── camera.py
│   ├── evaluate.py
│   └── servo_controller/
│       └── servo_controller.ino # ESP8266 Arduino sketch
│
├── init_project.py
├── requirements.txt
├── run_from_scratch.ps1         # Full reset + verification runbook
└── README.md
```

---

## 🐍 Python Version

```
Python 3.12.x
```

Create the venv with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: `mediapipe==0.10.9` only supports Python ≤3.11. On Python 3.12 use `mediapipe==0.10.21` (see `requirements.txt`).

---

## 🔧 Setup (Windows)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Or install manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install opencv-python "numpy<2" onnxruntime mediapipe==0.10.21 paho-mqtt
```

Initialize folder structure (creates `data/`, `dashboard/`, `models/`, etc.):

```powershell
python init_project.py
```

---

## 🧪 Automated Smoke Tests

Run non-interactive checks (imports, embedder, history, dashboard APIs, optional camera/MQTT):

```powershell
.\scripts\test_all.ps1
```

---

## 🔄 Full Reset Runbook

To wipe enrolled data and history, recreate folders, and verify the environment:

```powershell
.\run_from_scratch.ps1
```

This script auto-runs cleanup and checks (steps 0–9). Interactive pipeline steps (enroll, detect, dashboard, servo) are documented inside the script — uncomment or copy each command to run them one at a time.

---

## 🧠 ArcFace Model

Copy the model from the InsightFace buffalo pack:

```powershell
Copy-Item buffalo_l\w600k_r50.onnx models\embedder_arcface.onnx
```

---

## 👤 Step 1 — Enroll a Face

```powershell
python -m src.enroll
```

**Controls during enrollment:**

| Key | Action |
|-----|--------|
| `SPACE` | Capture one sample |
| `a` | Toggle auto-capture (every 0.25 s) |
| `s` | Save all samples to DB |
| `r` | Reset new samples (keep existing) |
| `q` | Quit |

- Needs **15 samples** by default (`EnrollConfig.samples_needed`)
- Saves aligned 112×112 crops to `data/enroll/<name>/`
- Stores the mean ArcFace embedding in `data/db/face_db.npz`

---

## ▶️ Step 2 — Run Face Locking (detect.py)

```powershell
python -m src.detect
```

Or pass the target name directly:

```powershell
python -m src.detect --name Alice
```

What this does:
1. Opens the camera (tries indices 0 → 1 → 2)
2. Prompts for a target identity (or uses `--name`) from the enrolled DB
3. Detects all faces each frame using `HaarFaceMesh5pt`
4. Embeds and matches every face against the DB (`dist_thresh=0.48` in `src/config.py`, plus enroll-crop matching and frame voting)
5. Locks onto the chosen target with identity matching + spatial fallback (≤ 50 px keypoint distance)
6. Releases lock after **2 seconds** of not seeing the target
7. Runs `ActionDetector` on the locked face — detects head movement (left/right/up/down) and smile
8. Logs all events via `HistoryManager` → `data/history/history_log.jsonl`
9. Writes live lock status to `data/history/lock_state.json` (read by the dashboard)

**Color coding:**

| Color | Meaning |
|-------|---------|
| 🟡 Yellow | Locked target face |
| 🟢 Green | Recognised other face |
| 🔴 Red | Unknown face |

Press `q` to quit.

---

## 📊 Step 2b — Live HTML Dashboard

In a **second terminal** (while `detect.py` is running):

```powershell
python -m src.dashboard
```

Optional flags: `--host 127.0.0.1` (default), `--port 8765` (default).

Open **http://127.0.0.1:8765** in your browser.

The dashboard shows:
- **Target person** — who `detect.py` is trying to lock onto
- **Lock status** — Locked / Searching / Idle
- **Movements** — head moves (`moved left`, `moved right`, etc.) for the locked person only
- **Event log** — full history (`LOCKED`, movements, `smile`, `UNLOCKED`)

Data is read from `data/history/history_log.jsonl` and `data/history/lock_state.json` (auto-refreshes every second).

**HTTP API endpoints** (used by the page; also useful for debugging):

| Route | Description |
|-------|-------------|
| `/` | Dashboard HTML (`dashboard/index.html`) |
| `/api/status` | Current lock state from `lock_state.json` |
| `/api/events?limit=200` | Recent events from `history_log.jsonl` |
| `/api/movements?limit=200` | Movement events only (`moved *`) |

---

## 🎯 Step 3 (Optional) — Run with Servo Control (faceLockServo.py)

```powershell
python -m src.faceLockServo
```

- Prompts you to choose a target identity from the enrolled DB
- Locks face and publishes **servo angles (0–180°)** to MQTT topic `TeAmSiX/facelocking/servo_ctrl_x9z`
- Logs **LOCKED**, head movements, **smile**, and **blink** to the same history files as `detect.py` (works with the dashboard)
- MQTT Broker: `157.173.101.159:1883`
- Uses **MediaPipe FaceMesh** (full 468 landmarks) for detection in this mode
- Servo angle is smoothed and rate-limited (deadzone: 5°, interval: 100 ms)

Run the dashboard in a second terminal while `faceLockServo` is active:

```powershell
python -m src.dashboard
```

---

## 🤖 ESP8266 Servo Controller

File: `src/servo_controller/servo_controller.ino`

- Connects to WiFi: `Main Hall`
- Subscribes to MQTT topic: `TeAmSiX/facelocking/servo_ctrl_x9z`
- Servo on pin `D1`, range 0–180°
- **Search mode**: if no MQTT message arrives for **1500 ms**, the servo sweeps back and forth automatically
- Smooth movement: increments target angle by 1° per `loop()` tick (15 ms delay)

**Dependencies (Arduino Library Manager):**
- `ESP8266WiFi`
- `PubSubClient`
- `Servo`

---

## 📝 Action Detection

`ActionDetector` runs on the locked face every frame using the 5-point keypoints `[left_eye, right_eye, nose, left_mouth, right_mouth]`:

| Action | Detection method |
|--------|-----------------|
| `moved left/right/up/down` | Bounding box centre displacement > 15 px |
| `smile` | Mouth width / face width ratio crosses 0.40 threshold |
| `blink` | Eye-to-nose distance drops then recovers (5-point proxy) |

Events are debounced (~0.9 s cooldown per action) and written immediately to `data/history/history_log.jsonl` (JSONL format).

**Recognition reliability:** matching uses the mean DB embedding plus all saved enroll crops, a lenient distance threshold (`0.48`), and temporal voting across frames so enrolled people stay recognized instead of flickering to Unknown.

---

## ❗ Common Errors

**`ModuleNotFoundError`**
- Make sure `src/__init__.py` exists
- Always run with `python -m src.detect` (not `python src/detect.py`)

**MediaPipe import error (Python 3.12)**
```powershell
pip install mediapipe==0.10.21
```

**`Failed to open camera`**
- `enroll.py` and `detect.py` try camera indices 0, 1, 2 automatically
- Make sure no other application is using the camera

**MQTT connection timeout**
- Check that the broker at `157.173.101.159:1883` is reachable from your network
- `faceLockServo.py` will still run in offline mode (servo commands will not be sent)

---

## 🔒 Changing the Lock Target

At runtime:

```powershell
python -m src.detect --name YourName
```

Or edit thresholds and paths in `src/config.py`.

---

## 🚀 Possible Extensions

- FAISS approximate nearest-neighbour search for large DBs
- Full blink detection (requires 68-point or FaceMesh EAR landmarks)
- WebSocket push instead of polling for the dashboard
- Multi-target locking
- Cloud/database event logging
