import cv2 as cv
import numpy as np
import imgprocess as ip
import time
import os
import argparse

# GENERAL PARAMETERS
SRCPATH = "flanges"
OUTPATH = "temp"
WINDOW_SCALE = 1/8
FONT_SIZE = 10
FONT_THICK = 20

# PROCESSING PARAMETERS
BLUR_KERNEL_SIZE = 9
MORPH_KERNEL_SIZE = (1, 1)
MORPH_PASSES = 1

# HOUGH CIRCLE PARAMETERS
ACCUMULATOR_RATIO = 1 # Ratio of image resolution to accumulator resolution; bigger = rougher circles detectable
ACCUMULATOR_THRESH = 50 # Higher = more likely to be true circle
CIRCLE_GAP = 30

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
        if (len(approx) > 8):
            circles.append(cv.fitEllipse(c))
    circles = sorted(circles, key = lambda rect: rect[1][0] * rect[1][1], reverse = True)
    return circles

def find_hCircle(img):
    (lower, upper) = canny_calc(img)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, ACCUMULATOR_RATIO, CIRCLE_GAP, param2 = ACCUMULATOR_THRESH, minRadius = 5, maxRadius = 500)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

def draw_contour(frame, contours):
    img_cnt = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    if contours is None:
        return img_cnt

    for c in contours:
        cv.ellipse(img_cnt, c, (0, 0, 255), 10)
    return img_cnt

def draw_hCircle(frame, hCircles):
    img_hough = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    if hCircles is None:
        return img_hough
    for i in hCircles[0, :]: # Unpacks 1xNx3 array into Nx3 array
        center = (i[0], i[1])
        cv.circle(img_hough, center, 1, (0, 0, 255), 3)
        radius = i[2]
        cv.circle(img_hough, center, radius, (0, 0, 255), 3)
    return img_hough

def crop(img, shapes): # This section needs fixing
    if shapes is None:
        return
    maxx = shapes[0]
    mask = np.ones((img.shape[0], img.shape[1]))
    cv.ellipse(mask, maxx, 1, thickness = -1)
##    rect = cv.boundingRect(maxx)
    print(mask)
    (cx, cy) = maxx[1] # Unpacks 2-element tuple
    return



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type = str, required = True, help = "Image index")
    args = vars(ap.parse_args())
    
    frame = cv.imread(os.path.join(SRCPATH, f"{args['image']}.jpg"))

    if frame is None:
        print("Unable to load picture")
        exit()

    # Blur before resizing to reduce artifacts - need to implement
    frame = ip.resizeImg(frame, width = 500)[0]

    original = frame.copy()
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    frame = cv.medianBlur(frame, BLUR_KERNEL_SIZE)
    # frame = cv.GaussianBlur(frame, (BLUR_KERNEL_SIZE,) * 2, 0)
    

    (lower, upper) = canny_calc(frame)
    frame = cv.Canny(frame, lower, upper)


    img_processed = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, MORPH_KERNEL_SIZE) # Morph functions aren't so useful for hollow objects
    frame = cv.morphologyEx(frame, cv.MORPH_OPEN, ker, MORPH_PASSES)

    contours = find_contour(frame)
    img_cnt = draw_contour(frame, contours)
    # crop(img_cnt, contours)

    hCircles = find_hCircle(frame)
    img_hough = draw_hCircle(frame, hCircles)
    
    frame = cv.hconcat((original, img_processed, img_cnt))

    labels = ("Original", "Processed", "Contoured")
    for (i, txt) in enumerate(labels):
        sz = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_THICK)
        cv.putText(frame, txt, (original.shape[1] * i, original.shape[0] - sz[1]), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 0), FONT_THICK)

    cv.namedWindow("Results", cv.WINDOW_NORMAL)
    scale = tuple([round(x * WINDOW_SCALE) for x in frame.shape[0:2]][::-1])
    cv.resizeWindow("Results", scale)
    cv.imshow("Results", frame)
    
    if cv.waitKey(0) == ord('s'):
        t = time.strftime("%Y%m%d-%H%M%S")
        cv.imwrite(os.path.join(OUTPATH, f"test_{t}.png"), frame)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

