# Automated smoke tests (no interactive camera windows).
# Run from project root:  .\scripts\test_all.ps1

Set-Location $PSScriptRoot\..

$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path

$fail = 0
function Test-Step($name, [scriptblock]$Block) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
        Write-Host "[FAIL] $name" -ForegroundColor Red
        $script:fail++
    } else {
        Write-Host "[PASS] $name" -ForegroundColor Green
    }
}

Test-Step "Python 3.12" { & $py --version }
Test-Step "init_project.py" { & $py init_project.py }
Test-Step "MediaPipe" { & $py -c "import mediapipe as mp; print(mp.__version__); mp.solutions.face_mesh.FaceMesh(); print('OK')" }
Test-Step "Packages" { & $py -c "import cv2, numpy, onnxruntime, paho.mqtt.client; print('cv2', cv2.__version__, 'numpy', numpy.__version__)" }
Test-Step "Module imports" { & $py -c "from src.config import DB_PATH; from src.detect import main; from src.enroll import main; from src.recognize import main; from src.faceLockServo import main; print('Imports OK')" }
Test-Step "ArcFace embedder" { & $py -c "import numpy as np; from src.embed import ArcFaceEmbedderONNX; e=ArcFaceEmbedderONNX(); r=e.embed(np.zeros((112,112,3),dtype=np.uint8)); print('dim', r.dim)" }
Test-Step "ActionDetector" { & $py -c "import numpy as np; from src.action_detector import ActionDetector; d=ActionDetector(); kps=np.array([[0,0],[10,0],[5,5],[2,10],[8,10]],dtype=np.float32); bbox=np.array([0,0,10,10],dtype=np.float32); d.update(kps,bbox); assert 'moved right' in d.update(kps+20, bbox+np.array([20,0,20,0]))" }
Test-Step "HistoryManager" { & $py -c "from src.history_manager import HistoryManager; HistoryManager().log_event('TEST','smoke')" }
Test-Step "Haar5ptDetector" { & $py -c "from src.haar_5pt import Haar5ptDetector; Haar5ptDetector()" }
Test-Step "FaceDBMatcher" { & $py -c "from src.recognize import load_db_npz, FaceDBMatcher; from src.config import DB_PATH; db=load_db_npz(DB_PATH) if DB_PATH.exists() else {}; FaceDBMatcher(db, 0.34); print('identities', len(db))" }
Test-Step "ArcFace model" { if (-not (Test-Path models\embedder_arcface.onnx)) { exit 1 } }
Test-Step "Camera probe" { & $py -c "import cv2; ok=any(cv2.VideoCapture(i).isOpened() for i in (0,1,2)); print('camera_available', ok); exit(0 if ok else 1)" }
Test-Step "evaluate.py" { & $py -m src.evaluate }
Test-Step "MQTT (localhost)" { & $py -m src.test_mqtt_connection }
Test-Step "Dashboard HTML" { if (-not (Test-Path dashboard\index.html)) { exit 1 } }
Test-Step "Dashboard APIs" {
    & $py -c "from src.dashboard import DashboardHandler; from src.lock_state import write_lock_state, read_lock_state; from src.history_manager import read_events_from_disk; write_lock_state(target_name='Test', is_locked=True, locked_name='Test', last_action='moved left'); assert read_lock_state()['target_name']=='Test'; print('OK')"
}

Write-Host "`n=== $fail test(s) failed ===" -ForegroundColor $(if ($fail -eq 0) { 'Green' } else { 'Red' })
exit $fail
