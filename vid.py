import cv2 as cv
import numpy as np
import matplotlib as plt

cap = cv.VideoCapture(0)

if not cap.isOpened(): # Only applicable if using VideoCapture();
    print("Unable to connect to camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not read")
        break

    disp = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    cv.imshow("View", disp)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
