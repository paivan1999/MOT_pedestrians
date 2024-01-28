import os

import cv2

from MOT_pedestrians import ROOT_DIR, ROOT_GLOB
from MOT_pedestrians.algorithms.detectors.Detectors import CompositionDetector
from MOT_pedestrians.algorithms.testing_detectors.test_flow import FlowWrapper


class VideoSaver:
    initial_cap = None
    flow_wrapper:FlowWrapper = None
    type = None
    @staticmethod
    def load(path):
        VideoSaver.initial_cap = cv2.VideoCapture(os.path.join(ROOT_DIR, path))
    @staticmethod
    def set_detector(detector):
        VideoSaver.flow_wrapper = FlowWrapper(detector)
    @staticmethod
    def draw(frame):
        VideoSaver.flow_wrapper.wrap(frame)
    @staticmethod
    def save(path):
        out = cv2.VideoWriter(os.path.join(ROOT_GLOB, path),cv2.VideoWriter.fourcc(*'mp4v'),
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
            cv2.imshow("main",frame)
            cv2.waitKey(1)
            n += 1
        out.release()
        VideoSaver.initial_cap.release()
