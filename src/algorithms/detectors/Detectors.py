import dataclasses
import random

from src.draw_instruments.mostly_different_colors import get_k_colors
import cv2
from abc import abstractmethod

import numpy as np


class BaseDetector:
    def __init__(self,**params):
        self.params = params

    @abstractmethod
    def detect(self,frame):
        pass

class DefaultBasedOnHOGDetector(BaseDetector):
    def __init__(self, hog_type:str = "DefaultPeopleDetector", winStride = (4, 4), **params):
        super().__init__(**params)
        self.hog = cv2.HOGDescriptor()
        if hog_type == "DefaultPeopleDetector":
            self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
            self.winStride = winStride
    def detect(self,frame):
        return self.hog.detectMultiScale(frame, winStride=self.winStride)

class DefaultBasedOnMultiTrackingDetector(BaseDetector):
    def __init__(self, max_count = 4, tracker_type:str = "CSRT", **params):
        super().__init__(**params)
        self.counter = max_count
        self.tracker = cv2.legacy.MultiTracker.create()
        self.detector = DefaultBasedOnHOGDetector(**params)
        if tracker_type == "CSRT":
            self.tracker_class = cv2.legacy.TrackerCSRT
    def add_trackers(self,frame):
        result = []
        for bbox,_ in zip(*self.detector.detect(frame)):
            self.tracker.add(self.tracker_class.create(), frame, tuple(bbox))
            result.append(bbox)
        return result
    def detect(self,frame):
        result = self.tracker.update(frame)[1]
        if self.counter > 0:
            result = list(result) + self.add_trackers(frame)
        self.counter -= 1
        return result

class CustomMultiTracker:
    def __init__(self):
        self.trackers_data = {}
        self.tracker_types = {"CSRT":cv2.legacy.TrackerCSRT}
        self.n = 0
    def __iter__(self):
        return self.trackers_data.items().__iter__()
    def add(self, frame, confidence, bbox, color, tracker_type="CSRT")->int:
        tracker = self.tracker_types[tracker_type].create()
        self.trackers_data[self.n] = [tracker, confidence, bbox, color]
        self.n += 1
        tracker.init(frame,bbox)
        return self.n - 1
    def replace(self, ind:int, tracker, confidence, bbox):
        self.trackers_data[ind] = [tracker, confidence, bbox, self.trackers_data[ind][3]]
    def pop(self, ind:int):
        self.trackers_data.pop(ind)
    def update_one(self, frame, confidence, bbox, ind: int, tracker_type="CSRT"):
        if confidence >= self.trackers_data[ind][1]:
            tracker = self.tracker_types[tracker_type].create()
            self.replace(ind,tracker,confidence,bbox)
            tracker.init(frame,bbox)
    def get_tracker(self,i):
        if i in self.trackers_data:
            return self.trackers_data[i][0]
        else:
            raise Exception("out of boundaries")
    def get_confidence(self,i):
        if i in self.trackers_data:
            return self.trackers_data[i][1]
        else:
            raise Exception("out of boundaries")
    def get_bbox(self,i):
        if i in self.trackers_data:
            return self.trackers_data[i][2]
        else:
            raise Exception("out of boundaries")
    def get_color(self,i):
        if i in self.trackers_data:
            return self.trackers_data[i][3]
        else:
            raise Exception("out of boundaries")
    def set_color(self,i,color):
        if i in self.trackers_data:
            self.trackers_data[i][3] = color
        else:
            raise Exception("out of boundaries")
    def update(self,frame):
        for_delete = []
        for ind,el in self:
            tracker = el[0]
            ret,el[2] = tracker.update(frame)
            if not ret:
                for_delete.append(ind)
        for ind in for_delete:
            self.pop(ind)

def diff_MAE(bbox1, bbox2)->float:
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2
    z1,q1 = x1+w1,y1+h1
    z2,q2 = x2+w2,y2+h2
    return abs(x1-x2) + abs(y1-y2) + abs(z1-z2) + abs(q1-q2)
def CommonArea(bbox1,bbox2)->float:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    z1, q1 = x1 + w1, y1 + h1
    z2, q2 = x2 + w2, y2 + h2
    left = max(x1, x2)
    top = min(q1, q2)
    right = min(z1, z2)
    bottom = max(y1, y2)
    width = right - left
    height = top - bottom
    if width < 0 or height < 0:
        return 0
    return width * height
def Area(bbox)->float:
    return bbox[2]*bbox[3]
def diff_CommonArea(bbox1, bbox2)->float:
    common_area = CommonArea(bbox1,bbox2)
    area1 = Area(bbox1)
    area2 = Area(bbox2)
    total = area1 + area2 - common_area
    return 1 - common_area/total
def diff_correctedCommonArea(bbox1,bbox2)->float:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    w = min(w1,w2)
    h = min(h1,h2)
    new_bbox1 = (x1 + (w1 - w)/2,y1 + (h1 - h)/2,w,h)
    new_bbox2 = (x2 + (w2 - w)/2,y2 + (h2 - h)/2,w,h)
    return diff_CommonArea(new_bbox1,new_bbox2)

def similar(bbox1,bbox2,threshold=0.22):
    return diff_correctedCommonArea(bbox1, bbox2) <= threshold
def most_similar_bbox(bbox1,bbox_list):
    smallest_diff = 10000000
    similar_numb = -1
    for i,bbox2 in enumerate(bbox_list):
        new_diff = diff_correctedCommonArea(bbox1, bbox2)
        if new_diff < smallest_diff:
            smallest_diff = new_diff
            similar_numb = i
    return (similar_numb,smallest_diff)

def filter_confidence(bbox_list,confidences,confidence_threshold):
    result_bbox_list = []
    result_confidences = []
    for bbox,confidence in zip(bbox_list,confidences):
        if confidence >= confidence_threshold:
            result_bbox_list.append(bbox)
            result_confidences.append(confidence)
    return result_bbox_list,result_confidences

def least_used_color(used_colors):
    result,_ = min(used_colors.items(),key=lambda el: el[1])
    return result
class CompositionDetector(BaseDetector):
    def __init__(
            self,
            detector:BaseDetector = DefaultBasedOnHOGDetector(),
            tracker_type:str = "CSRT",
            diff_threshold1 = 0.12,
            diff_threshold2 = 0.22,
            confidence_threshold = 0.7,
            colors_count = 7,
            not_bigger_than_by = 1.1,
            winStride = None,
            **params):
        self.customMultiTracker = CustomMultiTracker()
        if winStride:
            self.base_detector = DefaultBasedOnHOGDetector(winStride=winStride)
        else:
            self.base_detector = detector
        self.colors = [color for color in get_k_colors(colors_count + 1) if color != (255,255,255)]
        self.used_colors = {color:0 for color in self.colors}
        self.tracker_type = tracker_type
        self.diff_threshold1 = diff_threshold1
        self.diff_threshold2 = diff_threshold2
        self.confidence_threshold = confidence_threshold
        self.not_bigger_than_by = not_bigger_than_by
        super().__init__(**params)
    def detect(self,frame):
        base_bbox_list,base_confidences = self.base_detector.detect(frame)
        base_bbox_list,base_confidences = filter_confidence(base_bbox_list,base_confidences,self.confidence_threshold)
        self.customMultiTracker.update(frame)
        for ind,(_,confidence,bbox2,_) in self.customMultiTracker:
            smallest_diff = 1000000000000
            similar_box, similar_confidence = None,None
            for bbox1,base_confidence in zip(base_bbox_list,base_confidences):
                new_diff = diff_correctedCommonArea(bbox1, bbox2)
                if new_diff < smallest_diff:
                    smallest_diff = new_diff
                    similar_box = bbox1
                    similar_confidence = base_confidence
            if smallest_diff <= self.diff_threshold2:
                if similar_confidence >= confidence:
                    height2 = bbox2[3]
                    height1 = similar_box[3]
                    if height1/height2 <= self.not_bigger_than_by:
                        self.customMultiTracker.update_one(frame, similar_confidence, similar_box, ind,
                                                           tracker_type=self.tracker_type)
        for bbox1,base_confidence in zip(base_bbox_list,base_confidences):
            bboxes = [bbox2 for _,(_,_,bbox2,_) in self.customMultiTracker]
            if not bboxes or diff_correctedCommonArea(bbox1, min(bboxes, key = lambda el: diff_MAE(el, bbox1))) > self.diff_threshold2:
                color = least_used_color(self.used_colors)
                self.customMultiTracker.add(frame, base_confidence, bbox1, color)
                self.used_colors[color]+=1
        trackers_list = list(self.customMultiTracker.trackers_data.keys())
        trackers_pairs = [(a, b) for idx, a in enumerate(trackers_list) for b in trackers_list[idx + 1:]]
        deleted = set()
        for ind1,ind2 in trackers_pairs:
            if (ind1 in deleted or ind2 in deleted):
                continue
            if diff_correctedCommonArea(
                    self.customMultiTracker.get_bbox(ind1),
                    self.customMultiTracker.get_bbox(ind2)
            ) <= self.diff_threshold1:
                if self.customMultiTracker.get_confidence(ind1) < self.customMultiTracker.get_confidence(ind2):
                    if ind1 < ind2:
                        color_to_set = self.customMultiTracker.get_color(ind1)
                        color_to_replace = self.customMultiTracker.get_color(ind2)
                        self.used_colors[color_to_replace]-=1
                        self.used_colors[color_to_set]+=1
                        self.customMultiTracker.set_color(ind2, color_to_set)
                    self.used_colors[self.customMultiTracker.get_color(ind1)] -= 1
                    self.customMultiTracker.pop(ind1)
                    deleted.add(ind1)
                else:
                    if ind2 < ind1:
                        color_to_set = self.customMultiTracker.get_color(ind2)
                        color_to_replace = self.customMultiTracker.get_color(ind1)
                        self.used_colors[color_to_replace] -= 1
                        self.used_colors[color_to_set] += 1
                        self.customMultiTracker.set_color(ind1, color_to_set)
                    self.used_colors[self.customMultiTracker.get_color(ind2)] -= 1
                    self.customMultiTracker.pop(ind2)
                    deleted.add(ind2)
        return \
            [confidence for _, (_, confidence, _, _) in self.customMultiTracker], \
            [bbox for _,(_,_,bbox,_) in self.customMultiTracker],\
            [color for _,(_,_,_,color) in self.customMultiTracker]
