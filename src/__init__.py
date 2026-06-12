"""
Face recognition and face locking system.

This package implements the full pipeline:
  Camera -> Haar detection -> MediaPipe 5pt landmarks -> 112x112 align
  -> ArcFace embedding -> DB match -> lock / actions / optional MQTT servo

Main entry points (run from project root with venv active):
  python -m src.enroll          # register faces
  python -m src.detect          # face locking + action log
  python -m src.dashboard       # HTML event dashboard (run alongside detect)
  python -m src.faceLockServo   # locking + MQTT servo control
"""
