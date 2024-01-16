import cv2
import numpy as np
import pytest
from MOT_pedestrians.video_loaders.standard_video_loader import VideoLoader

@pytest.fixture
def video_loader():
    # Создаем временное видео для тестирования
    video_path = "test_video.mp4"
    create_test_video(video_path)
    loader = VideoLoader(video_path)
    yield loader
    # Удаляем временное видео после тестов
    import os
    if os.path.exists(video_path):
        os.remove(video_path)

def create_test_video(video_path):
    # Создаем временное видео с одним кадром
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 1.0, (width, height))
    out.write(frame)
    out.release()

def test_open_video(video_loader):
    assert video_loader.open_video()
    video_loader.release_video()


def test_read_frame(video_loader):
    video_loader.open_video()
    frame = video_loader.read_frame()
    assert frame is not None
    video_loader.release_video()

def test_get_frame_rate(video_loader):
    video_loader.open_video()
    frame_rate = video_loader.get_frame_rate()
    assert frame_rate == 1.0
    video_loader.release_video()

def test_get_frame_size(video_loader):
    video_loader.open_video()
    frame_size = video_loader.get_frame_size()
    assert frame_size == (100,100)
    video_loader.release_video()

# Дополнительный тест для release_video
def test_release_video(video_loader):
    video_loader.open_video()
    video_loader.release_video()
    # После release_video, видео не должно быть открыто
    assert not video_loader.cap.isOpened()