import cv2 as cv
import numpy as np
import matplotlib as plt
import imgprocess as ip
import time
import os
import sys

outpath = "temp"
if len(sys.argv) > 1:
    srcpath =os.path.join("flanges", f"{sys.argv[1]}.jpg")
else:
    sys.exit("No image index provided")

def main():
    cap = cv.imread(srcpath)

    if cap is None:
        print("Unable to load picture")
        exit()

    frame = cap

    frame = ip.resizeImg(frame, width = 500)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (11, 11), 0)
    #frame = cv.medianBlur(frame, 5)

    otsu_thresh, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Morphological open to isolate/denoise elliptical features
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel, iterations = 3)

    # Find contours
    cnts = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        pm = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * pm, True)
        area = cv.contourArea(c)
        if len(approx) > 5 and area > 1000 and area < 500000:
            ((x, y), r) = cv.minEnclosingCircle(c)
            cv.circle(frame, (int(x), int(y)), int(r), (36, 255, 12), 2)
    
    cv.imshow("View", frame)

    if cv.waitKey(0) == ord('s'):
        t = time.strftime("%Y%m%d-%H%M%S")
        cv.imwrite(os.path.join(outpath, f"test_{t}.png"), frame)
        

if __name__ == "__main__":
    main()
