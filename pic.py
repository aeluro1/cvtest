import cv2 as cv
import numpy as np
import matplotlib as plt
import imgprocess as ip
import time
import os
import argparse
srcpath = "flanges"
outpath = "temp"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, required = True, help = "Image index")
args = vars(ap.parse_args())

srcpath = os.path.join(srcpath, f"{args['image']}.jpg")

class Init:
    def __init__(self, init_: bool):
        self.init = init_
_init = Init(False)

def main():
    cap = cv.imread(srcpath)

    if cap is None:
        print("Unable to load picture")
        exit()

    frame = cap

    frame = ip.resizeImg(frame, width = 500)

    original = frame
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
##    frame = cv.GaussianBlur(frame, (5, 5), 0) # Review effect
    frame = cv.medianBlur(frame, 3)

##    frame = cv.bilateralFilter(frame, 5, 175, 175)
    frame = cv.Canny(frame, 75, 100)

    contours, h = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    circles = []
    for c in contours:
        approx = cv.approxPolyDP(c, 0.05 * cv.arcLength(c, True), True)
##        if ((len(approx) > 8) and (cv.isContourConvex(approx)) and (area > 3)):
##            circles.append(c);
        circles.append(c);
            
##            cv.imshow(original)
##            cv.waitKey(0)
    cv.drawContours(original, circles, -1, (0, 0, 255), -1)
            
##    # Morphological open to remove black noise blobs
##    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
##    frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel, iterations = 2)
##
##    # Find contours
##    cnts = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
##    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

##    for c in cnts:
##        pm = cv.arcLength(c, True)
##        approx = cv.approxPolyDP(c, 0.04 * pm, True)
##        area = cv.contourArea(c)
##        if len(approx) > 5 and area > 1000 and area < 500000:
##            ((x, y), r) = cv.minEnclosingCircle(c)
##            cv.circle(frame, (int(x), int(y)), int(r), (0, 0, 255), 2)

    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    
    frame = cv.hconcat((frame, original))
    cv.imshow("Results", frame)
    
    if cv.waitKey(0) == ord('s'):
        t = time.strftime("%Y%m%d-%H%M%S")
        cv.imwrite(os.path.join(outpath, f"test_{t}.png"), frame)

    cv.destroyAllWindows()
        

if __name__ == "__main__":
    main()
