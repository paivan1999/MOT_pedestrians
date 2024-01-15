import os

import cv2

from definitions import ROOT_DIR
from src.algorithms.detectors.Detectors import CompositionDetector


class VideoSaver:
    initial_cap = None
    detector = None
    type = None
    @staticmethod
    def load(path):
        VideoSaver.initial_cap = cv2.VideoCapture(os.path.join(ROOT_DIR, path))
    @staticmethod
    def set_detector(type, *args, **kwargs):
        VideoSaver.type = type
        if type == "CompositionDetector":
            VideoSaver.detector = CompositionDetector(*args, **kwargs)
    @staticmethod
    def draw(frame):
        if VideoSaver.type == "CompositionDetector":
            confidences, bbox_list, colors = VideoSaver.detector.detect(frame)
            for confidence, bbox, rgb in zip(confidences, bbox_list, colors):
                # for bbox in bboxes:
                x, y, w, h = [int(i) for i in bbox]
                bgr = tuple(reversed(rgb))
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
                cv2.putText(frame, str(round(confidence, 2)), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 2, 2)
    @staticmethod
    def save(path):
        out = cv2.VideoWriter(os.path.join(ROOT_DIR, path),cv2.VideoWriter.fourcc(*'mp4v'),
                              30,
                              (int(VideoSaver.initial_cap.get(3)),int(VideoSaver.initial_cap.get(4)))
                              )
        n = 1
        while True:
            if n % 50 == 0:
                print(f"\t\t\t{n}")
            ret,frame = VideoSaver.initial_cap.read()
            if not ret:
                break
            VideoSaver.draw(frame)
            out.write(frame)
            cv2.imshow("frame",frame)
            cv2.waitKey(1)
            n += 1
        out.release()
        VideoSaver.initial_cap.release()
