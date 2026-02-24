"""
camera.py — Camera capture module for Proppant QC.

Wraps OpenCV VideoCapture with auto-detection, resolution setting,
and fallback for systems without a camera.
"""
import cv2
import numpy as np


class CameraCapture:
    """USB/CSI camera wrapper using OpenCV."""

    def __init__(self):
        self.cap = None
        self.current_index = -1

    def open(self, index: int = 0, width: int = 1920, height: int = 1080) -> bool:
        """Open a camera by index. Returns True if successful."""
        self.release()
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.current_index = index
        return True

    def read_frame(self) -> np.ndarray | None:
        """Read one frame. Returns BGR numpy array or None."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.current_index = -1

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    @staticmethod
    def list_available(max_check: int = 5) -> list[int]:
        """Probe camera indices 0..max_check-1 and return available ones."""
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
