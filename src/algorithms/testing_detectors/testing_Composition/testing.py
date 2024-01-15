from src.algorithms.detectors.Detectors import CompositionDetector
import cv2
import os

from definitions import ROOT_DIR
from src.algorithms.testing_detectors.instruments import copy, get_new_file_name
from src.draw_instruments.instruments import showMovedWindow

detector = CompositionDetector(
    diff_threshold1=0.0,
    diff_threshold2=0.66,
    confidence_threshold=1.0,
    winStride=(8,8),
    colors_count=10,
    not_bigger_than_by=1.05
)

# Open the video file
cap = cv2.VideoCapture(os.path.join(ROOT_DIR,"data/people_on_street/POS3.mp4"))

showMovedWindow("main",-1,-1)
frames = []
n = 0
while True:
    # Read the next frame
    n += 1
    ret, frame = cap.read()

    if not ret:
        break

    # # Update the tracker with the new detections

    confidences, bbox_list, colors = detector.detect(frame)
    for confidence, bbox, rgb in zip(confidences,bbox_list,colors):
    # for bbox in bboxes:
        x, y, w, h = [int(i) for i in bbox]
        bgr = tuple(reversed(rgb))
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
        cv2.putText(frame,str(round(confidence,2)),(x,y - 3),cv2.FONT_HERSHEY_SIMPLEX,1,bgr,2,2)


    # Display the frame
    cv2.imwrite(os.path.join(ROOT_DIR,"examples\\example.jpg"),frame)
    cv2.imshow('main', frame)
    frame1 = frame
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
                source = "testing_Composition"
                destination = "Composition_examples"
                file_name = get_new_file_name("number.txt")
                cv2.imwrite(
                    os.path.join(ROOT_DIR, f"examples/images_for_VKR/{destination}/{file_name}"), frame)
                copy(os.path.join(ROOT_DIR, f"examples/images_for_VKR/{destination}/{file_name}"))
                start = False


# Release the resources
cap.release()
cv2.destroyAllWindows()