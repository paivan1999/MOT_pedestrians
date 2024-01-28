import time
from pathlib import Path

import os

import cv2

from MOT_pedestrians import ROOT_DIR
from MOT_pedestrians.algorithms.detectors.Detectors import CompositionDetector, DefaultBasedOnMultiTrackingDetector, \
    DefaultBasedOnHOGDetector
from MOT_pedestrians.algorithms.testing_detectors.instruments import get_new_file_name, copy
from MOT_pedestrians.draw_instruments.instruments import showMovedWindow


class FlowWrapper:
    def __init__(self,detector):
        self.detector = detector
    def wrap(self,frame):
        if isinstance(self.detector,DefaultBasedOnMultiTrackingDetector):
            bbox_list = self.detector.detect(frame)
            for bbox in bbox_list:
                x, y, w, h = [int(i) for i in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif isinstance(self.detector,DefaultBasedOnHOGDetector):
            bbox_list, confidences = self.detector.detect(frame)
            for confidence, bbox in zip(confidences, bbox_list):
                x, y, w, h = [int(i) for i in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(round(confidence, 2)), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            2)
        elif isinstance(self.detector,CompositionDetector):
            bbox_list, confidences, colors = self.detector.detect(frame)
            for confidence, bbox, rgb in zip(confidences, bbox_list, colors):
                # for bbox in bboxes:
                x, y, w, h = [int(i) for i in bbox]
                bgr = tuple(reversed(rgb))
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
                cv2.putText(frame, str(round(confidence, 2)), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 2, 2)
        cv2.imshow('main', frame)


class FlowInteractor:
    def __init__(self, save_dir, index_path):
        self.save_dir = save_dir
        self.index_path = index_path
    def interact(self, frame):
        key = cv2.waitKey(1)
        if key == ord('q'):
            return False
        elif key == ord('p'):
            start = True
            while True:
                key = cv2.waitKey(1)
                if key == ord('p'):
                    break
                elif key == ord('q'):
                    return False
                elif key == ord('c') and start:
                    file_name = get_new_file_name(self.index_path)
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    save_path = os.path.join(self.save_dir, file_name)
                    cv2.imwrite(save_path, frame)
                    copy(save_path)
                    start = False
        return True
def show_flow(wrapper:FlowWrapper, interactor:FlowInteractor, cap = None, path = os.path.join(ROOT_DIR,"data/people_on_street/POS3.mp4")):
    if path:
        cap = cv2.VideoCapture(path)
    elif not cap:
        raise Exception("you should pass video capture or path to video")
    showMovedWindow("main", -1, -1)
    t = 0
    n = 0
    while True:
        t1 = time.time()
        n += 1
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            break
        wrapper.wrap(frame)
        if not interactor.interact(frame):
            break
        t += time.time() - t1
        print(t/n)

def test_Based_On_HOG_Detector(
        cap = None,
        path = os.path.join(ROOT_DIR,"data/people_on_street/POS3.mp4"),
        index_path = "HOGDetector_index.txt",
        save_dir = os.path.join(ROOT_DIR,"MOT_pedestrians_examples/HOG_examples"),
        winStride = (8,8)
):
    wrapper = FlowWrapper(DefaultBasedOnHOGDetector(winStride=winStride))
    interactor = FlowInteractor(save_dir=save_dir,index_path=index_path)
    show_flow(wrapper=wrapper,interactor=interactor,cap=cap,path=path)

def test_Based_On_MultiTracker_Detector(
        cap = None,
        path=os.path.join(ROOT_DIR, "data/people_on_street/POS3.mp4"),
        index_path="MultiTrackerDetector_index.txt",
        save_dir=os.path.join(ROOT_DIR, "MOT_pedestrians_examples/MultiTracker_examples"),
        max_count=1,
        winStride=(8,8)
):
    wrapper = FlowWrapper(DefaultBasedOnMultiTrackingDetector(max_count=max_count,winStride=winStride))
    interactor = FlowInteractor(save_dir=save_dir, index_path=index_path)
    show_flow(wrapper=wrapper, interactor=interactor, cap=cap, path=path)

def test_CompositionDetector(
        cap = None,
        path=os.path.join(ROOT_DIR, "data/people_on_street/POS3.mp4"),
        index_path="CompositionDetector_index.txt",
        save_dir=os.path.join(ROOT_DIR, "MOT_pedestrians_examples/Composition_examples"),
        diff_threshold1=0.0,
        diff_threshold2=0.66,
        confidence_threshold=1.0,
        winStride=(8, 8),
        colors_count=16,
        not_bigger_than_by=1.05
):
    wrapper = FlowWrapper(CompositionDetector(
        diff_threshold1=diff_threshold1,
        diff_threshold2=diff_threshold2,
        confidence_threshold=confidence_threshold,
        winStride=winStride,
        colors_count=colors_count,
        not_bigger_than_by=not_bigger_than_by
    ))
    interactor = FlowInteractor(save_dir=save_dir, index_path=index_path)
    show_flow(wrapper=wrapper, interactor=interactor, cap=cap, path=path)

