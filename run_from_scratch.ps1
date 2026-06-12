# =============================================================================
# Face Locking — Full test runbook (start from clean data)
# =============================================================================
#
# HOW TO USE:
#   1. Open PowerShell in the project folder (this repo root).
#   2. You may run this whole script, OR copy/paste sections one at a time.
#   3. Interactive steps (enroll, detect, servo) need you at the keyboard.
#
# Always keep (.venv) active before running Python commands.
# =============================================================================


# -----------------------------------------------------------------------------
# STEP 0 — Go to project folder
# -----------------------------------------------------------------------------
# What it does: Ensures all relative paths (data/, models/) resolve correctly.
Set-Location $PSScriptRoot


# -----------------------------------------------------------------------------
# STEP 1 — Activate the virtual environment
# -----------------------------------------------------------------------------
# What it does: Uses Python 3.12 + packages from .venv (NOT system Python 3.13).
# You should see (.venv) in your prompt after this.
.\.venv\Scripts\Activate.ps1


# -----------------------------------------------------------------------------
# STEP 2 — Wipe all enrolled people, database, and history
# -----------------------------------------------------------------------------
# What each line removes:
#   data\enroll\*       — face crop images per person (data/enroll/<name>/*.jpg)
#   data\db\*           — face_db.npz + face_db.json (recognition database)
#   data\history\*      — history_log.jsonl + lock_state.json (event log + live status)
#   data\debug_aligned\*— saved alignment test snapshots
#   history\*           — legacy history folder (old code, if present)
#   debug_output.txt    — MediaPipe debug log
#
# Does NOT delete: models\embedder_arcface.onnx (keep the AI model).
Remove-Item -Recurse -Force data\enroll\*       -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force data\db\*           -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force data\history\*      -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force data\debug_aligned\*  -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force history\*             -ErrorAction SilentlyContinue
Remove-Item -Force debug_output.txt               -ErrorAction SilentlyContinue

Write-Host "[OK] All enrolled data and history cleared." -ForegroundColor Green


# -----------------------------------------------------------------------------
# STEP 3 — Recreate empty folder structure
# -----------------------------------------------------------------------------
# What it does: Runs init_project.py to create data/enroll, data/db,
#               data/history, dashboard/, data/debug_aligned, models/, etc.
python init_project.py


# -----------------------------------------------------------------------------
# STEP 4 — Verify data folders are empty
# -----------------------------------------------------------------------------
# What it does: Lists everything under data/. Should show empty dirs only.
Get-ChildItem -Recurse data


# -----------------------------------------------------------------------------
# STEP 5 — Check Python version (must be 3.12 from .venv)
# -----------------------------------------------------------------------------
# What it does: Confirms you are NOT accidentally on Python 3.13.
python --version
where.exe python


# -----------------------------------------------------------------------------
# STEP 6 — Check MediaPipe (required for face landmarks)
# -----------------------------------------------------------------------------
# What it does: Verifies mediapipe imports and mp.solutions.face_mesh works.
# Expected output: version (e.g. 0.10.21) and "OK". Warnings are harmless.
python -c "import mediapipe as mp; print(mp.__version__); mp.solutions.face_mesh.FaceMesh(); print('OK')"


# -----------------------------------------------------------------------------
# STEP 7 — Check required packages
# -----------------------------------------------------------------------------
# What it does: Confirms opencv, numpy, onnxruntime, mediapipe, paho-mqtt exist.
pip show opencv-python numpy onnxruntime mediapipe paho-mqtt


# -----------------------------------------------------------------------------
# STEP 8 — Check ArcFace model and dashboard files exist
# -----------------------------------------------------------------------------
# What it does: Returns True if models\embedder_arcface.onnx and dashboard\index.html exist.
# If False for the model, copy your model first, e.g.:
#   Copy-Item buffalo_l\w600k_r50.onnx models\embedder_arcface.onnx
Test-Path models\embedder_arcface.onnx
Test-Path dashboard\index.html


# -----------------------------------------------------------------------------
# STEP 9 — Import smoke test (no camera)
# -----------------------------------------------------------------------------
# What it does: Loads main project modules without opening the webcam.
python -c "from src.config import DB_PATH, LOCK_STATE_PATH, DASHBOARD_DIR; from src.detect import main; from src.enroll import main; from src.dashboard import main; from src.lock_state import write_lock_state, read_lock_state; print('Imports OK')"


# -----------------------------------------------------------------------------
# STEP 9b — Automated smoke tests (optional, no interactive windows)
# -----------------------------------------------------------------------------
# What it does: Runs scripts/test_all.ps1 — embedder, history, dashboard APIs,
#               evaluate.py, MQTT ping, camera probe, etc.
# .\scripts\test_all.ps1


# =============================================================================
# PIPELINE TESTS — copy/paste ONE block at a time (press q to quit each window)
# Interactive commands are commented out (#) so this script only auto-runs
# cleanup + checks (steps 0-9). Remove the leading # to run a step.
# =============================================================================


# -----------------------------------------------------------------------------
# STEP 10 — src/camera.py — Camera only
# -----------------------------------------------------------------------------
# What it does: Opens webcam (tries indices 0, 1, 2). No face detection.
# Pass: live video window appears.
# python -m src.camera


# -----------------------------------------------------------------------------
# STEP 11 — src/landmarks.py — Haar + 5 landmark dots
# -----------------------------------------------------------------------------
# What it does: Draws green Haar box and 5 green dots on face landmarks.
# python -m src.landmarks


# -----------------------------------------------------------------------------
# STEP 12 — src/haar_5pt.py — Smoothed detector (enrollment-style)
# -----------------------------------------------------------------------------
# What it does: Same detector family as enroll; stable box + 5 points.
# python -m src.haar_5pt


# -----------------------------------------------------------------------------
# STEP 13 — src/align.py — Face alignment to 112x112
# -----------------------------------------------------------------------------
# What it does: Two windows — camera with landmarks + aligned face preview.
# Optional: press s to save snapshot to data/debug_aligned/
# python -m src.align


# -----------------------------------------------------------------------------
# STEP 14 — src/embed.py — ArcFace embedding demo
# -----------------------------------------------------------------------------
# What it does: Shows embedding heatmap and stats for detected face.
# Optional: press p to print embedding values in terminal.
# python -m src.embed


# -----------------------------------------------------------------------------
# STEP 15 — src/enroll.py — Enroll a new person (REQUIRED before detect)
# -----------------------------------------------------------------------------
# What it does:
#   - Prompts for a name (e.g. Alice)
#   - SPACE = capture sample, a = auto-capture, s = save to DB, q = quit
#   - Needs ~15 samples, then press s to write data/db/face_db.npz
# python -m src.enroll
#
# After enrolling, verify DB and crops:
# Test-Path data\db\face_db.npz
# Get-ChildItem data\enroll


# -----------------------------------------------------------------------------
# STEP 16 — src/recognize.py — Multi-face recognition (no locking)
# -----------------------------------------------------------------------------
# What it does: Green = known face, Red = unknown. Keys: r reload, +/- threshold.
# python -m src.recognize


# -----------------------------------------------------------------------------
# STEP 17 — src/evaluate.py — Threshold tuning (optional)
# -----------------------------------------------------------------------------
# What it does: Analyzes enrolled crops; suggests best dist_thresh value.
# Needs at least 5 crops per person in data/enroll/<name>/.
# python -m src.evaluate


# -----------------------------------------------------------------------------
# STEP 18 — src/detect.py — Face locking + action history
# -----------------------------------------------------------------------------
# What it does:
#   - Locks onto the named person (yellow box)
#   - Logs LOCKED, movements, smile to data/history/history_log.jsonl
#   - Writes live status to data/history/lock_state.json (for the dashboard)
# Replace Alice with the name you enrolled in step 15.
# python -m src.detect --name Alice
#
# After quitting, view the event log and lock state:
# Get-Content data\history\history_log.jsonl
# Get-Content data\history\lock_state.json


# -----------------------------------------------------------------------------
# STEP 18b — src/dashboard.py — Live HTML dashboard (run with detect.py)
# -----------------------------------------------------------------------------
# What it does:
#   - Serves dashboard/index.html at http://127.0.0.1:8765
#   - Polls lock_state.json + history_log.jsonl every second
# Start detect.py (step 18) in one terminal, then run this in another:
# python -m src.dashboard
#
# Optional: custom port
# python -m src.dashboard --port 8765
#
# Verify dashboard files exist:
# Test-Path dashboard\index.html
# Test-Path data\history\lock_state.json


# -----------------------------------------------------------------------------
# STEP 19 — src/faceLockServo.py — MQTT servo control (optional)
# -----------------------------------------------------------------------------
# What it does: Locks face + sends servo angles to ESP8266 via MQTT.
# Also logs LOCKED, movements, smile, blink (same files as detect.py).
# Run dashboard in a second terminal: python -m src.dashboard
# Requires broker reachable and Arduino flashed with servo_controller.ino.
# python -m src.faceLockServo


# -----------------------------------------------------------------------------
# STEP 20 — src/test_mqtt_connection.py — MQTT ping only (optional)
# -----------------------------------------------------------------------------
# What it does: Tests connection to the MQTT broker without camera.
# python -m src.test_mqtt_connection


# =============================================================================
# QUICK REFERENCE — files and what they do
# =============================================================================
#
#   init_project.py      Create empty data/, dashboard/, and models/ folders
#   requirements.txt     Pinned Python dependencies (Python 3.12)
#   scripts/test_all.ps1 Automated smoke tests (imports, APIs, optional camera)
#   run_from_scratch.ps1 This runbook — reset data + verify environment
#   dashboard/index.html Live dashboard UI (served by dashboard.py)
#   src/config.py        Shared paths, thresholds, MQTT settings
#   src/camera.py        Webcam test only
#   src/landmarks.py     Haar detection + 5-point landmarks demo
#   src/haar_5pt.py      Smoothed Haar + FaceMesh detector demo
#   src/align.py         5-point alignment to 112x112 demo
#   src/embed.py         ArcFace embedding visualization demo
#   src/enroll.py        Capture faces and build face_db.npz
#   src/recognize.py     Recognize all faces in frame (no lock)
#   src/evaluate.py      Tune recognition threshold from enroll crops
#   src/detect.py        Lock onto one person + log actions + lock_state.json
#   src/dashboard.py     HTTP server for live HTML dashboard
#   src/lock_state.py    Read/write data/history/lock_state.json
#   src/faceLockServo.py Lock + MQTT servo angle control
#   src/action_detector.py   Movement/smile detection (used by detect)
#   src/history_manager.py   Writes history_log.jsonl; read_events_from_disk()
#   src/servo_controller/servo_controller.ino   ESP8266 firmware
#
# =============================================================================

Write-Host "`n[DONE] Cleanup + checks finished (steps 0-9)." -ForegroundColor Cyan
Write-Host "Open run_from_scratch.ps1 and run steps 10-20 one at a time (uncomment or copy each command)." -ForegroundColor Cyan
Write-Host "For automated checks only, run: .\scripts\test_all.ps1" -ForegroundColor Cyan
