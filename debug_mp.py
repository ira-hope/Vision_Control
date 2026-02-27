
import sys

def log(msg):
    with open("debug_output.txt", "a") as f:
        f.write(str(msg) + "\n")

try:
    log("Starting debug script")
    import mediapipe
    log(f"Mediapipe version: {getattr(mediapipe, '__version__', 'unknown')}")
    log(f"Mediapipe file: {mediapipe.__file__}")
    log(f"Dir mediapipe: {dir(mediapipe)}")
    
    try:
        import mediapipe.solutions
        log("Successfully imported mediapipe.solutions")
    except ImportError as e:
        log(f"Failed to import mediapipe.solutions: {e}")

    if hasattr(mediapipe, 'solutions'):
        log("mediapipe.solutions exists")
    else:
        log("mediapipe.solutions DOES NOT exist")

except ImportError as e:
    log(f"Failed to import mediapipe: {e}")
except Exception as e:
    log(f"Unexpected error: {e}")
