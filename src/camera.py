import cv2

def main():
    cap = None
    for _idx in (0, 1, 2):
        _cap = cv2.VideoCapture(_idx)
        if _cap.isOpened():
            cap = _cap
            print(f"Camera opened on index {_idx}.")
            break
        _cap.release()

    if not cap:
        raise RuntimeError("Camera not opened. Try changing index (0/1/2).")

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
