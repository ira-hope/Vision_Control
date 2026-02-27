import cv2
import time

def test_all_cameras():
    """Test all possible camera indices"""
    print("Testing cameras 0 through 10...")
    
    for i in range(10):  # Try up to 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"\nCamera {i} opened successfully!")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ Can read frames: {frame.shape}")
                
                # Try to show the frame
                cv2.imshow(f"Camera {i}", frame)
                print("  Press any key to continue...")
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            else:
                print(f"  ✗ Can't read frames from camera {i}")
            
            cap.release()
        else:
            print(f"Camera {i}: Not available")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_all_cameras()
    input("\nPress Enter to exit...")
