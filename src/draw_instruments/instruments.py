import pyautogui
import cv2
screensize = pyautogui.size().width,pyautogui.size().height

def showMovedWindow(winname, x, y, screen_sized = True, w = 900,h = 900, p = 0.85):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    if screen_sized:
        cv2.resizeWindow(winname, int(screensize[0]*p),int(screensize[1]*p))
    else:
        cv2.resizeWindow(winname, w, h)