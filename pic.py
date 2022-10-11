import cv2 as cv
import numpy as np
import imgprocess as ip
import time
import os
import argparse

SRCPATH = "flanges"
OUTPATH = "temp"

BLUR_KERNEL_SIZE = 5

# Calculate Canny parameters based on median intensity of filtered image
# https://stackoverflow.com/questions/41893029/opencv-canny-edge-detection-not-working-properly
def canny_calc(img, s = 0.33):
    v = np.median(img)
    if v > 191: # light images
        th1 = int(max(0, (1.0 - 2 * s) * (255 - v)))
        th2 = int(max(85, (1.0 + 2 * s) * (255 - v)))
    elif v > 127:
        th1 = int(max(0, (1.0 - s) * (255 - v)))
        th2 = int(max(255, (1.0 + s) * (255 - v)))
    elif v < 63: # dark images
        th1 = int(max(0, (1.0 - 2 * s) * v))
        th2 = int(max(85, (1.0 + 2 * s) * v))
    else:
        th1 = int(max(0, (1.0 - s) * v))
        th2 = int(max(85, (1.0 + s) * v))
    return (th1, th2)

def find_contour(img):
    contours, h = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    circles = []
    for c in contours:
        approx = cv.approxPolyDP(c, 0.02 * cv.arcLength(c, True), False) # Closed contour boolean set to false as many holes are open; (cv.isContourConvex(approx)) and (area > 3) also removed
        if (True):#(len(approx) > 8)
            circles.append(c);
    return circles

def find_hCircle(img):
    (lower, upper) = canny_calc(img)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1 = upper, param2 = 75, minRadius = 0, maxRadius = 0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type = str, required = True, help = "Image index")
    args = vars(ap.parse_args())
    
    frame = cv.imread(os.path.join(SRCPATH, f"{args['image']}.jpg"))

    if frame is None:
        print("Unable to load picture")
        exit()

    frame = ip.resizeImg(frame, width = 500)

    original = frame.copy()
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.medianBlur(frame, BLUR_KERNEL_SIZE)
    
    (lower, upper) = canny_calc(frame)
    frame = cv.Canny(frame, lower, upper)

    contours = find_contour(frame)
    img_cnt = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    img_cnt = cv.drawContours(img_cnt, contours, -1, (0, 0, 255), -1)

    hCircles = find_hCircle(frame)
    img_hough = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    for i in hCircles[0, :]: # Unpacks 1xNx3 array into Nx3 array
        center = (i[0], i[1])
        cv.circle(img_hough, center, 1, (0, 0, 255), 3)
        
        radius = i[2]
        cv.circle(img_hough, center, radius, (0, 0, 255), 3)
    
    frame = cv.hconcat((original, img_hough, img_cnt))
    cv.imshow("Results", frame)
    
    if cv.waitKey(0) == ord('s'):
        t = time.strftime("%Y%m%d-%H%M%S")
        cv.imwrite(os.path.join(OUTPATH, f"test_{t}.png"), frame)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
