"""
Haar face detection + MediaPipe FaceMesh 5-point landmarks.

Keypoints (FaceMesh indices): L_eye(33), R_eye(263), nose(1), L_mouth(61), R_mouth(291)

Classes:
  Haar5ptDetector   — single largest face, EMA-smoothed (used by enroll, embed, align)
  HaarFaceMesh5pt   — multi-face, one mesh per Haar ROI (used by detect, recognize)
  align_face_5pt()  — warp to ArcFace 112x112 template

Run demo:  python -m src.haar_5pt
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e


# -------------------------
# Data
# -------------------------
@dataclass
class FaceKpsBox:
    """One detected face: axis-aligned box, score, and 5 landmark points (5,2)."""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32


# -------------------------
# Helpers
# -------------------------
def _estimate_norm_5pt(
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Build 2x3 affine matrix that maps your 5pts to ArcFace-style template.
    kps order must be: [Leye, Reye, Nose, Lmouth, Rmouth]
    """
    k = kps_5x2.astype(np.float32)

    # ArcFace 112x112 template (InsightFace standard)
    dst = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041],  # right mouth
        ],
        dtype=np.float32,
    )

    out_w, out_h = int(out_size[0]), int(out_size[1])

    if (out_w, out_h) != (112, 112):
        sx = out_w / 112.0
        sy = out_h / 112.0
        dst = dst * np.array([sx, sy], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(k, dst, method=cv2.LMEDS)

    if M is None:
        M = cv2.getAffineTransform(
            np.array([k[0], k[1], k[2]], dtype=np.float32),
            np.array([dst[0], dst[1], dst[2]], dtype=np.float32),
        )

    return M.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (aligned_bgr, M)
    """
    M = _estimate_norm_5pt(kps_5x2, out_size=out_size)
    out_w, out_h = int(out_size[0]), int(out_size[1])

    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, M


def _clip_box_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    bb = b.astype(np.float32).copy()
    bb[0] = np.clip(bb[0], 0, W - 1)
    bb[1] = np.clip(bb[1], 0, H - 1)
    bb[2] = np.clip(bb[2], 0, W - 1)
    bb[3] = np.clip(bb[3], 0, H - 1)
    return bb


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a face bbox from 5 keypoints with asymmetric padding.
    """
    k = kps.astype(np.float32)
    x_min, x_max = float(k[:, 0].min()), float(k[:, 0].max())
    y_min, y_max = float(k[:, 1].min()), float(k[:, 1].max())

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    return np.array(
        [
            x_min - pad_x * w,
            y_min - pad_y_top * h,
            x_max + pad_x * w,
            y_max + pad_y_bot * h,
        ],
        dtype=np.float32,
    )


def _ema(
    prev: Optional[np.ndarray],
    cur: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float = 12.0) -> bool:
    """
    Quick sanity filter on 5pt geometry.
    """
    le, re, no, lm, rm = kps.astype(np.float32)

    if np.linalg.norm(re - le) < min_eye_dist:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False

    return True


# -------------------------
# Detector
# -------------------------
class Haar5ptDetector:
    """
    Single-face detector with temporal smoothing (EMA on box and keypoints).

    Best for enrollment and demos where only one person is in frame.
  """

    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.80,
        debug: bool = True,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = float(smooth_alpha)

        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe==0.10.9"
            )

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def _facemesh_5pt(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark
        idxs = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE,
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT,
        ]

        kps = np.array(
            [[lm[i].x * W, lm[i].y * H] for i in idxs],
            dtype=np.float32,
        )

        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 1) -> List[FaceKpsBox]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            return []

        areas = faces[:, 2] * faces[:, 3]
        i = int(np.argmax(areas))
        x, y, w, h = faces[i].tolist()

        kps = self._facemesh_5pt(frame_bgr)
        if kps is None:
            if self.debug:
                print("[haar_5pt] Haar face but FaceMesh failed")
            return []

        margin = 0.35
        inside = (
            (kps[:, 0] >= x - margin * w)
            & (kps[:, 0] <= x + (1 + margin) * w)
            & (kps[:, 1] >= y - margin * h)
            & (kps[:, 1] <= y + (1 + margin) * h)
        )
        if inside.mean() < 0.60:
            return []

        if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * w)):
            return []

        box = _bbox_from_5pt(kps)
        box = _clip_box_xyxy(box, W, H)

        box_s = _ema(self._prev_box, box, self.smooth_alpha)
        kps_s = _ema(self._prev_kps, kps, self.smooth_alpha)

        self._prev_box = box_s.copy()
        self._prev_kps = kps_s.copy()

        x1, y1, x2, y2 = box_s.tolist()

        return [
            FaceKpsBox(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                score=1.0,
                kps=kps_s.astype(np.float32),
            )
        ][:max_faces]


# Alias used by detect.py / recognize.py (same fields as FaceKpsBox)
FaceDet = FaceKpsBox


class HaarFaceMesh5pt:
    """
    Multi-face detector: Haar finds candidates, FaceMesh runs per ROI.

    Returns up to max_faces FaceKpsBox instances per frame.
    Used by detect.py and recognize.py for simultaneous multi-person scenes.
    """

    def __init__(
        self,
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):
        if mp is None:
            raise RuntimeError(_MP_IMPORT_ERROR)

        self.debug = debug
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.min_size = min_size
        self.idxs = [33, 263, 1, 61, 291]

    def detect(self, frame: np.ndarray, max_faces: int = 5) -> List[FaceKpsBox]:
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=self.min_size
        )

        out: List[FaceKpsBox] = []
        for (x, y, w, h) in faces[:max_faces]:
            roi = frame[y : y + h, x : x + w]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark
            kps = np.array(
                [[lm[i].x * w + x, lm[i].y * h + y] for i in self.idxs],
                dtype=np.float32,
            )

            if not _kps_span_ok(kps, max(10.0, 0.18 * w)):
                continue

            bb = _bbox_from_5pt(kps)
            bb = _clip_box_xyxy(bb, W, H)
            x1, y1, x2, y2 = (int(round(v)) for v in bb.tolist())

            out.append(FaceKpsBox(x1, y1, x2, y2, 1.0, kps))

        return out


# -------------------------
# Demo
# -------------------------
def main():
    from .config import open_camera

    try:
        cap = open_camera()
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=True)

    print("Haar + 5pt (FaceMesh). Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        faces = det.detect(frame)

        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        else:
            cv2.putText(
                vis,
                "no face",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        cv2.imshow("haar_5pt", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
