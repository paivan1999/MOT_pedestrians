import cv2
import os

from definitions import ROOT_DIR
from src.draw_instruments.instruments import showMovedWindow
from src.algorithms.detectors.Detectors import DefaultBasedOnMultiTrackingDetector, DefaultBasedOnHOGDetector

from src.algorithms.testing_detectors.instruments import copy
from src.algorithms.testing_detectors.instruments import get_new_file_name

# Load the pre-trained HOG detector
detector = DefaultBasedOnMultiTrackingDetector(max_count=4,winStride=(8,8))
# Open the video file
cap = cv2.VideoCapture(os.path.join(ROOT_DIR,"data/people_on_street/POS2.mov"))

showMovedWindow("main",-1,-1)

while True:
    # Read the next frame
    ret, frame = cap.read()

    if not ret:
        break

    bbox_list = detector.detect(frame)
    for bbox in bbox_list:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

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
                source = "testing_based_on_MultiTracker"
                destination = "MultiTracker_examples"
                file_name = get_new_file_name("number.txt")
                cv2.imwrite(
                    os.path.join(ROOT_DIR, f"examples/images_for_VKR/{destination}/{file_name}"), frame)
                copy(os.path.join(ROOT_DIR, f"examples/images_for_VKR/{destination}/{file_name}"))
                start = False

# Release the resources
cap.release()
cv2.destroyAllWindows()