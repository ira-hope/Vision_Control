import numpy as np

class ActionDetector:
    def __init__(self):
        self.prev_kps = None
        self.prev_center = None
        self.prev_eye_open = None
        self.prev_mouth_open = None
        
        # Thresholds
        self.move_thr = 15.0
        self.eye_closed_thr = 0.015  # Based on 5pt alignment relative measures if avail, 
                                     # but here we use simple 5pt geometry approximations
        self.smile_thr = 0.40        # Mouth width / Face width

    def update(self, kps: np.ndarray, bbox: np.ndarray) -> list[str]:
        """
        kps: (5, 2) landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
        bbox: [x1, y1, x2, y2]
        Returns: list of detected action strings e.g. ["blink", "moved left"]
        """
        actions = []
        
        # 1. Movement
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        curr_center = (cx, cy)
        
        if self.prev_center is not None:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]
            
            dirs = []
            if dx > self.move_thr: dirs.append("right")
            elif dx < -self.move_thr: dirs.append("left")
            
            if dy > self.move_thr: dirs.append("down")
            elif dy < -self.move_thr: dirs.append("up")
            
            if dirs:
                actions.append("moved " + "/".join(dirs))
        
        self.prev_center = curr_center

        # Geometry helpers
        # kps order: 0:L_eye, 1:R_eye, 2:Nose, 3:L_mouth, 4:R_mouth
        k = kps.astype(np.float32)
        
        # 2. Blink (crude approximation from 5 points is hard, usually need 68 or specific eye contour)
        # However, we can track vertical distance if available, BUT with only center points (pupils),
        # blink detection is NOT reliable. 
        # The user's previous code had `eye_open` attribute, likely from `detect.py` logic which was 
        # trying to read attributes that didn't exist in `FaceDet`.
        # Real blink detection needs full mesh or specific landmarks.
        # Since we use `HaarFaceMesh5pt` which uses MediaPipe FaceMesh, we DO have access to full mesh *internally*,
        # but `detect` only returns 5 points.
        
        # For now, we will assume we can't reliably detect blink from just these 5 points 
        # unless we change `haar_5pt.py` to return more data (like EAR).
        # Let's check `haar_5pt.py` again. It returns `kps` (5,2).
        
        # Wait, the user's previous `detect.py` code had:
        # eye_open = getattr(focus_face, "eye_open", 0.0)
        # This confirms it wasn't working before because `FaceDet` didn't have `eye_open`.
        
        # For this task, I will stick to what's possible with 5 points or movement provided.
        # We can detect Smile from 5 points (Mouth width).
        
        # 3. Smile
        # Mouth Width / Face Width
        face_width = bbox[2] - bbox[0]
        mouth_width = np.linalg.norm(k[4] - k[3]) # L_mouth to R_mouth
        
        ratio = mouth_width / max(1.0, face_width)
        
        if self.prev_mouth_open is not None:
            # Simple hysteresis or just threshold crossing
            if self.prev_mouth_open < self.smile_thr and ratio >= self.smile_thr:
                actions.append("smile")
                
        self.prev_mouth_open = ratio
        
        return actions
