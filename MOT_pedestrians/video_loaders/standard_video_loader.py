import cv2
from typing import Optional, Tuple
import numpy as np
cv2.HOGDescriptor.getDefaultPeopleDetector()
class VideoLoader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
    def open_video(self) -> bool:
        self.cap = cv2.VideoCapture(self.video_path)
        return self.cap.isOpened()
    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    def get_frame_rate(self) -> Optional[float]:
        if self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return None
    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        if self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return None
    def release_video(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
def resize_frame(frame: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
def save_frame(frame: np.ndarray, output_path: str) -> None:
    cv2.imwrite(output_path, frame)