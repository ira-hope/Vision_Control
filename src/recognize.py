"""
Multi-face recognition and database matching.

Pipeline: Haar -> FaceMesh 5pt (per ROI) -> align -> ArcFace -> cosine distance

Exports:
  FaceDBMatcher, load_db_npz  — used by detect.py and faceLockServo.py
  recognize_face()            — lazy singleton API for single-image recognition

Run demo:  python -m src.recognize
Keys: q quit | r reload DB | +/- adjust distance threshold
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .action_detector import ActionDetector
from .config import (
    DB_PATH,
    DISTANCE_THRESHOLD,
    ENROLL_DIR,
    MATCH_MARGIN,
    MIN_RECOGNITION_SIMILARITY,
    RECOGNITION_VOTE_WINDOW,
    open_camera,
)
from .embed import ArcFaceEmbedderONNX
from .haar_5pt import HaarFaceMesh5pt, align_face_5pt


# -------------------------
# Data
# -------------------------
@dataclass
class MatchResult:
    """Result of matching one embedding against the enrolled database."""
    name: Optional[str]       # matched name, or None if below threshold
    distance: float           # cosine distance (1 - similarity)
    similarity: float         # cosine similarity in [0, 1]
    accepted: bool            # True if distance <= dist_thresh


# -------------------------
# Lazy single-image recognizer API
# -------------------------
_DET: Optional[HaarFaceMesh5pt] = None
_EMBEDDER: Optional[ArcFaceEmbedderONNX] = None
_MATCHER: Optional["FaceDBMatcher"] = None


def _lazy_init_singleton() -> None:
    global _DET, _EMBEDDER, _MATCHER

    if _DET is None:
        _DET = HaarFaceMesh5pt()
    if _EMBEDDER is None:
        _EMBEDDER = ArcFaceEmbedderONNX()
    if _MATCHER is None:
        db = load_db_npz(DB_PATH)
        _MATCHER = FaceDBMatcher(db, dist_thresh=DISTANCE_THRESHOLD)


def recognize_face(face_img) -> Tuple[str, float]:
    """Recognize a single face crop. Returns (name, similarity)."""
    _lazy_init_singleton()
    assert _DET is not None and _EMBEDDER is not None and _MATCHER is not None

    faces = _DET.detect(face_img, max_faces=1)
    if not faces:
        return "Unknown", 0.0

    f = faces[0]
    aligned, _ = align_face_5pt(face_img, f.kps)
    res = _EMBEDDER.embed(aligned)
    mr = _MATCHER.match(res.embedding)

    name = mr.name if mr.name is not None else "Unknown"
    return name, mr.similarity


# -------------------------
# Math helpers
# -------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


# -------------------------
# DB helpers
# -------------------------
def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    return {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}


def load_enroll_sample_embeddings(
    enroll_dir: Path,
    embedder: ArcFaceEmbedderONNX,
) -> Dict[str, List[np.ndarray]]:
    """Load embeddings from saved 112x112 enroll crops (best-effort)."""
    samples: Dict[str, List[np.ndarray]] = {}
    if not enroll_dir.exists():
        return samples

    for person_dir in sorted(enroll_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        embs: List[np.ndarray] = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            try:
                embs.append(embedder.embed(img).embedding)
            except Exception:
                continue
        if embs:
            samples[person_dir.name] = embs
    return samples


# -------------------------
# Matcher
# -------------------------
class FaceDBMatcher:
    """Nearest-neighbour matcher over L2-normalized ArcFace embeddings."""

    def __init__(
        self,
        db: Dict[str, np.ndarray],
        dist_thresh: float,
        sample_embeddings: Optional[Dict[str, List[np.ndarray]]] = None,
    ):
        self.db = db
        self.dist_thresh = dist_thresh
        self.sample_embeddings = sample_embeddings or {}
        self.names = sorted(db.keys())
        self.mat = (
            np.stack([db[n] for n in self.names]).astype(np.float32)
            if self.names
            else None
        )

    @classmethod
    def from_disk(
        cls,
        db_path: Path = DB_PATH,
        dist_thresh: float = DISTANCE_THRESHOLD,
        enroll_dir: Path = ENROLL_DIR,
        embedder: Optional[ArcFaceEmbedderONNX] = None,
    ) -> "FaceDBMatcher":
        db = load_db_npz(db_path)
        samples: Dict[str, List[np.ndarray]] = {}
        if embedder is not None:
            samples = load_enroll_sample_embeddings(enroll_dir, embedder)
        return cls(db, dist_thresh, samples)

    def reload(self, path: Path):
        self.__init__(load_db_npz(path), self.dist_thresh, self.sample_embeddings)

    def _identity_similarity(self, emb: np.ndarray, name: str) -> float:
        sims = [float(np.dot(self.db[name], emb.reshape(-1)))]
        for sample in self.sample_embeddings.get(name, []):
            sims.append(float(np.dot(sample.reshape(-1), emb.reshape(-1))))
        return max(sims)

    def match(self, emb: np.ndarray) -> MatchResult:
        """Return best DB match with lenient rules for enrolled identities."""
        if not self.names:
            return MatchResult(None, 1.0, 0.0, False)

        emb = emb.reshape(-1).astype(np.float32)
        sims = np.array(
            [self._identity_similarity(emb, name) for name in self.names],
            dtype=np.float32,
        )
        i = int(np.argmax(sims))
        sim = float(sims[i])
        dist = 1.0 - sim
        name = self.names[i]

        second_sim = 0.0
        if len(sims) > 1:
            sims_copy = sims.copy()
            sims_copy[i] = -1.0
            second_sim = float(np.max(sims_copy))
        margin = sim - second_sim

        ok = (
            dist <= self.dist_thresh
            or sim >= MIN_RECOGNITION_SIMILARITY
            or (margin >= MATCH_MARGIN and sim >= MIN_RECOGNITION_SIMILARITY - 0.05)
        )

        return MatchResult(name if ok else None, dist, sim, ok)


class RecognitionStabilizer:
    """Temporal voting so enrolled faces do not flicker to Unknown."""

    def __init__(
        self,
        enrolled_names: set[str],
        window: int = RECOGNITION_VOTE_WINDOW,
        track_radius_px: float = 80.0,
    ):
        self.enrolled_names = enrolled_names
        self.window = window
        self.track_radius_px = track_radius_px
        self._tracks: Dict[int, Deque[tuple[Optional[str], float, bool]]] = {}
        self._centers: Dict[int, np.ndarray] = {}
        self._next_id = 0

    def _assign_track(self, center: np.ndarray) -> int:
        best_id = None
        best_dist = self.track_radius_px

        for tid, prev_center in self._centers.items():
            dist = float(np.linalg.norm(center - prev_center))
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
            self._tracks[best_id] = deque(maxlen=self.window)

        self._centers[best_id] = center
        return best_id

    def resolve(
        self,
        kps: np.ndarray,
        candidate_name: Optional[str],
        similarity: float,
        accepted: bool,
    ) -> tuple[str, bool]:
        center = kps.mean(axis=0)
        tid = self._assign_track(center)
        history = self._tracks[tid]
        history.append((candidate_name, similarity, accepted))

        voted = Counter(
            name for name, sim, ok in history
            if name in self.enrolled_names and (ok or sim >= MIN_RECOGNITION_SIMILARITY - 0.04)
        )
        if voted:
            winner = voted.most_common(1)[0][0]
            return winner, True

        if accepted and candidate_name:
            return candidate_name, True

        best_name, best_sim, _ = max(history, key=lambda item: item[1])
        if best_name in self.enrolled_names and best_sim >= MIN_RECOGNITION_SIMILARITY - 0.06:
            return best_name, True

        return "Unknown", False


# -------------------------
# Demo
# -------------------------
def main():
    det = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher.from_disk(DB_PATH, embedder=embedder)
    stabilizer = RecognitionStabilizer(set(matcher.names))
    detectors: Dict[str, ActionDetector] = {}

    cap = open_camera()
    print("Recognize: q quit | r reload | +/- threshold")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        faces = det.detect(frame)

        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps)
            res = embedder.embed(aligned)
            mr = matcher.match(res.embedding)
            name, accepted = stabilizer.resolve(
                f.kps, mr.name, mr.similarity, mr.accepted,
            )
            color = (0, 255, 0) if accepted else (0, 0, 255)

            action_text = ""
            if accepted:
                if name not in detectors:
                    detectors[name] = ActionDetector()

                bbox = np.array([f.x1, f.y1, f.x2, f.y2])
                actions = detectors[name].update(f.kps, bbox)
                if actions:
                    action_text = f" [{actions[-1]}]"

            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            cv2.putText(
                vis,
                f"{name} d={mr.distance:.3f}{action_text}",
                (f.x1, f.y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("recognize", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            matcher.reload(DB_PATH)
            print("DB reloaded")
        elif key in (ord("+"), ord("=")):
            matcher.dist_thresh += 0.01
            print("thr =", matcher.dist_thresh)
        elif key == ord("-"):
            matcher.dist_thresh -= 0.01
            print("thr =", matcher.dist_thresh)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
