"""
Scan camera indices 0–9 and show a preview for each working device.

Run:  python -m src.testcamera

Useful when open_camera() fails and you need to find which index works.
"""

import cv2
import time


def test_all_cameras():
    """Try indices 0–9; show one frame from each opened camera."""
    print("Testing cameras 0 through 10...")

    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"\nCamera {i} opened successfully!")

            ret, frame = cap.read()
            if ret:
                print(f"  Can read frames: {frame.shape}")
                cv2.imshow(f"Camera {i}", frame)
                print("  Press any key to continue...")
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            else:
                print(f"  Cannot read frames from camera {i}")

            cap.release()
        else:
            print(f"Camera {i}: Not available")

    print("\nTest complete!")


if __name__ == "__main__":
    test_all_cameras()
    input("\nPress Enter to exit...")
