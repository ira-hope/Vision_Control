"""
Face enrollment — build data/db/face_db.npz from live camera samples.

Pipeline: camera -> Haar5ptDetector -> align 112x112 -> ArcFace embedding

Run:  python -m src.enroll
Keys: SPACE capture | a auto-capture | s save to DB | r reset new samples | q quit

Output:
  data/enroll/<name>/*.jpg   aligned crop images
  data/db/face_db.npz        mean embedding per person
  data/db/face_db.json       metadata
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    DB_JSON_PATH,
    DB_PATH,
    ENROLL_DIR,
    ENROLL_SAMPLES_DEFAULT,
    ENROLL_SAMPLES_MAX,
    ENROLL_SAMPLES_MIN,
    open_camera,
)
from .embed import ArcFaceEmbedderONNX
from .haar_5pt import Haar5ptDetector, align_face_5pt


# -------------------------
# Config
# -------------------------
@dataclass
class EnrollConfig:
    out_db_npz: Path = DB_PATH
    out_db_json: Path = DB_JSON_PATH
    save_crops: bool = True
    crops_dir: Path = ENROLL_DIR
    samples_needed: int = ENROLL_SAMPLES_DEFAULT
    samples_min: int = ENROLL_SAMPLES_MIN
    samples_max: int = ENROLL_SAMPLES_MAX
    auto_capture_every_s: float = 0.25
    max_existing_crops: int = 300

    window_main: str = "enroll"
    window_aligned: str = "aligned_112"


# -------------------------
# DB helpers
# -------------------------
def ensure_dirs(cfg: EnrollConfig) -> None:
    cfg.out_db_npz.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_db_json.parent.mkdir(parents=True, exist_ok=True)

    if cfg.save_crops:
        cfg.crops_dir.mkdir(parents=True, exist_ok=True)


def load_db(cfg: EnrollConfig) -> Dict[str, np.ndarray]:
    if cfg.out_db_npz.exists():
        data = np.load(cfg.out_db_npz, allow_pickle=True)
        return {k: data[k].astype(np.float32) for k in data.files}
    return {}


def save_db(cfg: EnrollConfig, db: Dict[str, np.ndarray], meta: dict) -> None:
    ensure_dirs(cfg)
    np.savez(
        cfg.out_db_npz,
        **{k: v.astype(np.float32) for k, v in db.items()},
    )
    cfg.out_db_json.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    E = np.stack([e.reshape(-1) for e in embeddings], axis=0).astype(np.float32)
    m = E.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)
    return m.astype(np.float32)


# -------------------------
# Crops loader
# -------------------------
def _list_existing_crops(person_dir: Path, max_count: int) -> List[Path]:
    if not person_dir.exists():
        return []

    files = sorted(p for p in person_dir.glob("*.jpg") if p.is_file())
    if len(files) > max_count:
        files = files[-max_count:]
    return files


def load_existing_samples_from_crops(
    cfg: EnrollConfig,
    emb: ArcFaceEmbedderONNX,
    person_dir: Path,
) -> List[np.ndarray]:
    if not cfg.save_crops:
        return []

    crops = _list_existing_crops(person_dir, cfg.max_existing_crops)
    base: List[np.ndarray] = []

    for p in crops:
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            r = emb.embed(img)
            base.append(r.embedding)
        except Exception:
            continue

    return base


# -------------------------
# UI helpers
# -------------------------
def draw_status(
    frame: np.ndarray,
    name: str,
    base_count: int,
    new_count: int,
    needed: int,
    auto: bool,
    msg: str = "",
) -> None:
    total = base_count + new_count
    lines = [
        f"ENROLL: {name}",
        f"Existing: {base_count} | New: {new_count} | Total: {total} (need {needed}, min 10, max 30)",
        f"Auto: {'ON' if auto else 'OFF'} (toggle: a)",
        "SPACE=capture | s=save | r=reset NEW | q=quit",
    ]

    if msg:
        lines.insert(0, msg)

    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 26


# -------------------------
# Main enrollment loop
# -------------------------
def main():
    cfg = EnrollConfig()
    ensure_dirs(cfg)

    name = input("Enter person name to enroll (e.g., Alice): ").strip()
    if not name:
        print("No name provided. Exiting.")
        return

    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)
    emb = ArcFaceEmbedderONNX(debug=False)

    db = load_db(cfg)
    person_dir = cfg.crops_dir / name

    if cfg.save_crops:
        person_dir.mkdir(parents=True, exist_ok=True)

    base_samples = load_existing_samples_from_crops(cfg, emb, person_dir)
    new_samples: List[np.ndarray] = []

    status_msg = ""
    if base_samples:
        status_msg = f"Loaded {len(base_samples)} existing samples from disk."

    auto = False
    last_auto = 0.0

    cap = open_camera()

    cv2.namedWindow(cfg.window_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(cfg.window_aligned, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window_aligned, 240, 240)

    print("\nEnrollment started.")
    print("Controls: SPACE=capture, a=auto, s=save, r=reset NEW, q=quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            vis = frame.copy()
            faces = det.detect(frame, max_faces=1)
            aligned: Optional[np.ndarray] = None

            if faces:
                f = faces[0]
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                cv2.imshow(cfg.window_aligned, aligned)
            else:
                cv2.imshow(cfg.window_aligned, np.zeros((112, 112, 3), np.uint8))

            now = time.time()
            if auto and aligned is not None and (now - last_auto) >= cfg.auto_capture_every_s:
                r = emb.embed(aligned)
                new_samples.append(r.embedding)
                last_auto = now
                status_msg = f"Auto captured NEW ({len(new_samples)})"

                if cfg.save_crops:
                    cv2.imwrite(str(person_dir / f"{int(now * 1000)}.jpg"), aligned)

            draw_status(
                vis,
                name=name,
                base_count=len(base_samples),
                new_count=len(new_samples),
                needed=cfg.samples_needed,
                auto=auto,
                msg=status_msg,
            )

            cv2.imshow(cfg.window_main, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto = not auto
            if key == ord("r"):
                new_samples.clear()
            if key == ord(" ") and aligned is not None:
                r = emb.embed(aligned)
                new_samples.append(r.embedding)

            if key == ord("s"):
                all_samples = base_samples + new_samples
                if len(all_samples) < cfg.samples_min:
                    status_msg = (
                        f"Need at least {cfg.samples_min} samples "
                        f"(have {len(all_samples)})"
                    )
                    continue
                if len(all_samples) > cfg.samples_max:
                    all_samples = all_samples[: cfg.samples_max]
                    status_msg = f"Using first {cfg.samples_max} samples"
                template = mean_embedding(all_samples)
                db[name] = template

                meta = {
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_dim": int(template.size),
                    "names": sorted(db.keys()),
                }

                save_db(cfg, db, meta)
                print(f"Saved '{name}' to DB")

                base_samples = load_existing_samples_from_crops(cfg, emb, person_dir)
                new_samples.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
