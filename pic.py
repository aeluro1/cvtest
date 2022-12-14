import argparse
import os
import time

import cv2 as cv
import numpy as np

import imgprocess as ip

# GENERAL PARAMETERS
SRCPATH = "flanges"
OUTPATH = "temp"
WINDOW_SCALE = 1
CONTOUR_FIT_MIN = 50        # Minimum ellipse confidence %
CONTOUR_FIT_MAX = 150

# PROCESSING PARAMETERS - TUNE FOR DIFFERENT CAMERA
RESIZE_WIDTH = 500
BLUR_KERNEL_SIZE = 5
MORPH_KERNEL_SIZE = (1, 1)
MORPH_PASSES = 1

FONT_SCALE = RESIZE_WIDTH / 1000
FONT_THICK = 1

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
    
    ellipses = []
    for c in contours:
        clen = cv.arcLength(c, False)
        approx = cv.approxPolyDP(c, 0.02 * clen, False) # Closed contour boolean set to false as many holes are open; (cv.isContourConvex(approx)) and (area > 3) also removed
        if (len(approx) > 8):
            e = cv.fitEllipse(c)
            diff = clen / ip.ellipseCirc(e) * 100
            print(diff)
            if (CONTOUR_FIT_MIN < diff < CONTOUR_FIT_MAX):
                ellipses.append(e)
            # ellipses.append(e)
    circles = sorted(ellipses, key = lambda rect: rect[1][0] * rect[1][1], reverse = True)
    return circles

def draw_contour(frame, contours):
    img_cnt = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    if contours is None:
        return img_cnt

    for c in contours:
        cv.ellipse(img_cnt, c, (0, 0, 255), 2)
    return img_cnt
    
def filter_results(contours):
    maxdim = max(contours[0][1])
    (cx, cy) = contours[0][0]
    hbound = (cx - maxdim, cx + maxdim)
    vbound = (cy - maxdim, cy+ maxdim)
    print(hbound)
    print(vbound)
    for cnt in contours:
        if hbound[0] < cnt[0][0] < hbound[1] and vbound[0] < cnt[0][1] < vbound[1]:
            contours.remove(cnt)
    return contours



def process(frame):
    if frame is None:
        print("Unable to load picture")
        exit()
    
    frame = ip.resizeImg(frame, width = RESIZE_WIDTH)[0]

    frame = cv.medianBlur(frame, BLUR_KERNEL_SIZE)
    # frame = cv.GaussianBlur(frame, (BLUR_KERNEL_SIZE,) * 2, 0)
    original = frame.copy()
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    (lower, upper) = canny_calc(frame)
    frame = cv.Canny(frame, lower, upper)


    img_processed = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    # ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, MORPH_KERNEL_SIZE) # Morph functions aren't so useful for hollow objects
    # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, ker, MORPH_PASSES)

    contours = find_contour(frame)
    #contours = filter_results(contours)
    img_cnt = draw_contour(frame, contours)
    # crop(img_cnt, contours)
    
    frame = cv.hconcat((original, img_processed, img_cnt))
    
    return frame

def show(frame, vid = False):
    labels = ("Original", "Processed", "Contoured")
    for (i, txt) in enumerate(labels):
        sz = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, thickness = FONT_THICK)
        (x, y) = int(frame.shape[1] / 3 * i), frame.shape[0] - sz[1]
        cv.putText(frame, txt, (x, y), cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICK)
    
    cv.namedWindow("Results", cv.WINDOW_NORMAL)
    scale = tuple([round(x * WINDOW_SCALE) for x in frame.shape[0:2]][::-1])
    cv.resizeWindow("Results", scale)
    cv.imshow("Results", frame)
    
    if vid:
        k = cv.waitKey(1)
        if k == ord('s'):
            pass
        elif k != -1:
            return -1
    else:
        k = cv.waitKey(0)
        if k == ord('s'):
            t = time.strftime("%Y%m%d-%H%M%S")
            cv.imwrite(os.path.join(OUTPATH, f"test_{t}.png"), frame)
    return 0
    


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type = str, required = True, help = "Image index")
    args = vars(ap.parse_args()) # Convert from argparse.Namespace object to dictionary
    
    frame = cv.imread(os.path.join(SRCPATH, f"{args['image']}.jpg"))
    frame = process(frame)
    show(frame)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

