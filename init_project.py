from pathlib import Path

# Canonical project structure (relative to project root)
structure = {
    "data/enroll": [],
    "data/db": [],
    "data/history": [],
    "dashboard": [
        "index.html",
    ],
    "data/debug_aligned": [],
    "models": [
        "embedder_arcface.onnx",
    ],
    "src": [
        "__init__.py",
        "config.py",
        "camera.py",
        "detect.py",
        "landmarks.py",
        "align.py",
        "embed.py",
        "enroll.py",
        "recognize.py",
        "evaluate.py",
        "haar_5pt.py",
        "faceLockServo.py",
        "action_detector.py",
        "history_manager.py",
    ],
    "book": [],
}

for folder, files in structure.items():
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_path = folder_path / file
        if not file_path.exists():
            file_path.touch()

print("Face locking project structure verified.")
