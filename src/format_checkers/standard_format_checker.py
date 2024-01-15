import cv2
from typing import Optional

class VideoFormatChecker:
    @staticmethod
    def check_video_format(video_path: str) -> Optional[str]:
        """
        Проверяет формат видео по расширению файла.

        :param video_path: Путь к видеофайлу.
        :return: Строка с форматом видео (например, 'mp4', 'avi', 'mkv') или None, если формат неизвестен.
        """
        _, file_extension = video_path.rsplit('.', 1)
        return file_extension.lower() if file_extension else None

    @staticmethod
    def get_video_resolution(video_path: str) -> Optional[tuple]:
        """
        Получает разрешение видео.

        :param video_path: Путь к видеофайлу.
        :return: Кортеж (ширина, высота) видео или None, если разрешение неизвестно.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()
        return width, height

    @staticmethod
    def get_video_duration(video_path: str) -> Optional[float]:
        """
        Получает продолжительность видео в секундах.

        :param video_path: Путь к видеофайлу.
        :return: Продолжительность видео в секундах или None, если продолжительность неизвестна.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()
        return frame_count / fps if fps > 0 else None