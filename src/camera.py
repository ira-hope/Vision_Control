"""
Webcam smoke test — no face detection.

Run:  python -m src.camera
Quit: press q

Use this first to confirm the camera opens (indices 0, 1, 2 are tried).
"""

import cv2

from .config import open_camera


def main():
    # Open first available camera index
    cap = open_camera()

    print("Camera test. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        cv2.imshow("Camera Test", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
