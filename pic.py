import cv2 as cv
import numpy as np
import matplotlib as plt
import imgprocess as ip
import time
import os
import argparse
outpath = "temp"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, required = True, help = "Image index")
args = vars(ap.parse_args())
##import sys
##
##if len(sys.argv) > 1:
##    srcpath =os.path.join("flanges", f"{sys.argv[1]}.jpg")
##else:
##    sys.exit("No image index provided")

srcpath = os.path.join("flanges", f"{args['image']}.jpg")

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
    frame = cv.medianBlur(frame, 5)

    #(T, thresh) = cv.threshold(frame, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 2)
    frame = cv.bitwise_not(frame)

    contours, h = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for c in contours:
        print(len(c))
        print(c.shape)
        approx = cv.approxPolyDP(c, 0.03 * cv.arcLength(c, True), True)
        if len(c) > 10 and cv.isContourConvex(approx):
            cv.drawContours(original, [c], 0, (0, 0, 255), -1)
            
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
    cv.imshow("View", frame)
    cv.imshow("Circled", original)
    if cv.waitKey(0) == ord('s'):
        t = time.strftime("%Y%m%d-%H%M%S")
        frame = cv.hconcat((frame, original))
        cv.imwrite(os.path.join(outpath, f"test_{t}.png"), frame)

    cv.destroyAllWindows()
        

if __name__ == "__main__":
    main()
