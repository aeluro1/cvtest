import argparse
import os

import cv2 as cv

import pic

# GENERAL PARAMETERS
SRCPATH = "flanges"
FPS_OUT = 60

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
    while ret:
        result = pic.process(frame)
        if pic.show(result, True) == -1:
            break
        ret, frame = cap.read()

    cv.destroyAllWindows()
    cap.release()



def save():
# choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('video.avi', fourcc, OUT_FPS, (width, height), True)

    for j in range(0,5):
        img = cv2.imread(str(i) + '.png')
        video.write(img)
    video.release()

if __name__ == "__main__":
    main()