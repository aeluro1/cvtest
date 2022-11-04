import argparse
import os

import cv2 as cv

import pic

# GENERAL PARAMETERS
SRCPATH = "flanges"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--index", type = str, required = True, help = "Video index")
    args = vars(ap.parse_args()) # Convert from argparse.Namespace object to dictionary
    vidpath = os.path.join(SRCPATH, f"{args['index']}.mov")
    
    cap = cv.VideoCapture(vidpath)
    
    if not cap.isOpened():
        print("Unable to connect to camera")
        exit()

    ret, frame = cap.read()
    while ret is not None:
        if pic.process(frame) == -1:
            break
        ret, frame = cap.read()

    cv.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
