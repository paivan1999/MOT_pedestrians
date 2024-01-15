import cv2
import os

from definitions import ROOT_DIR
from src.draw_instruments.instruments import showMovedWindow
from src.algorithms.detectors.Detectors import DefaultBasedOnHOGDetector

from src.algorithms.testing_detectors.instruments import copy
from src.algorithms.testing_detectors.instruments import get_new_file_name

# Load the pre-trained HOG detector
detector = DefaultBasedOnHOGDetector(winStride=(8,8))
# Open the video file
cap = cv2.VideoCapture(os.path.join(ROOT_DIR,"data/people_on_street/POS2.mov"))

showMovedWindow("main",-1,-1)

def mean(l):
    return sum(l)/len(l)

def stdr(l):
    m = mean(l)
    return (sum([(el - m)**2 for el in l])/len(l))**(1/2)
counter = 1
means = []
stdrs = []
while True:
    # Read the next frame
    ret, frame = cap.read()

    if not ret:
        break
    if counter % 10 == 0:
        print(f"mean = {mean(means)}")
        print(f"stdr = {mean(stdrs)}")
    counter += 1
    bbox_list, confidences = detector.detect(frame)
    means.append(mean(confidences))
    stdrs.append(stdr(confidences))
    for confidence,bbox in zip(confidences,bbox_list):
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(frame, str(round(confidence, 2)), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 2)

    # Display the frame
    cv2.imshow('main', frame)
    # Exit if the user presses 'q' key
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        start = True
        while True:
            key = cv2.waitKey(1)
            if key == ord('p'):
                break
            elif key == ord('q'):
                quit()
            elif key == ord('c') and start:
                source = "testing_based_on_HOG"
                destination = "based_on_HOG_examples"
                file_name = get_new_file_name("number.txt")
                cv2.imwrite(
                    os.path.join(ROOT_DIR, f"examples/images_for_VKR/{destination}/{file_name}"), frame)
                copy(os.path.join(ROOT_DIR, f"examples/images_for_VKR/{destination}/{file_name}"))
                start = False

# Release the resources
cap.release()
cv2.destroyAllWindows()

# import cv2
# import os
#
# from definitions import ROOT_DIR
# from src.draw_instruments.instruments import showMovedWindow
# # Load the pre-trained HOG detector
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
#
# # Create a tracker object
# tracker = cv2.legacy.MultiTracker.create()
#
# # Open the video file
# cap = cv2.VideoCapture(os.path.join(ROOT_DIR,"data/people_on_street/POS2.mov"))
#
# showMovedWindow("main",-1,-1)
# # Loop over each frame of the video
# # ret, frame = cap.read()
# #
# # if not ret:
# #     raise Exception("can't read first frame")
# #
# # # Detect people in the frame
# # bbox_list, _ = hog.detectMultiScale(frame, winStride=(8, 8))
# #
# # # Update the tracker with the new detections
# # for bbox in bbox_list:
# #     tracker.add(cv2.legacy.TrackerMIL.create(), frame, tuple(bbox))
# counter = 0
# while True:
#     # Read the next frame
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # # Update the tracker with the new detections
#     if counter < 40:
#         bbox_list, confidences = hog.detectMultiScale(frame, winStride=(2, 2))
#         for bbox in bbox_list:
#             tracker.add(cv2.legacy.TrackerCSRT.create()., frame, tuple(bbox))
#     counter += 1
#     # Draw the bounding boxes for each tracked person
#     success, bboxes = tracker.up
#     if success:
#         for bbox, confidence in zip(bbox_list,confidences):
#         # for bbox in bboxes:
#             x, y, w, h = [int(i) for i in bbox]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame,str(round(confidence,2)),(x,y - 3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,2)
#
#     # Display the frame
#     cv2.imshow('main', frame)
#     # Exit if the user presses 'q' key
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
#
# # Release the resources
# cap.release()
# cv2.destroyAllWindows()